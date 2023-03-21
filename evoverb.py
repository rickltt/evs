from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.data_utils import InputFeatures
from openprompt import Verbalizer
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger
import random
import copy

class EvoVerbalizer(Verbalizer):
    r"""
    Args:
        num_candidates (:obj:`int`, optional): the number of candidates for further selection based on Section 4.1
        label_word_num_per_class (:obj:`int`, optional): set to be greater than 1 to support Multi-Verbalizers in Section 4.2
        max_it (:obj:`int`, optional): Maximnum number of label_words search. After reaching this number, the verbalizer will use the same label_words as the previous iterations.
        popsize (:obj:`int`, optional): the id of current search, used to determine when to stop label words searching.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer = None,
                 num_candidates: Optional[int]= 1000,
                 popsize: Optional[int] = 30,
                 max_it: Optional[int] = 3,
                 pm: Optional[float] = 0.2,
                 pc: Optional[float] = 0.8,
                 label_word_num_per_class: Optional[int] = 5,
                 num_classes: Optional[bool] = None,
                 classes: Optional[List[str]] = None,
                # init_using_split: Optional[str] = "train",
                 **kwargs):
        super().__init__(num_classes=num_classes, tokenizer = tokenizer, classes=classes)
        self.num_candidates = num_candidates
        self.popsize = popsize
        self.max_it = max_it
        self.label_word_num_per_class = label_word_num_per_class
        self.probs_buffer, self.labels_buffer = None, None
        self.pm, self.pc = pm, pc
        self.accumulate_step = 0 # currently not used, to support not epoch-level optimize.
        self.accumulate = True # A flag to indicate whether to
                               # accumulate examples for optimization.
                               # set to False after finish optimization.
        self.mats = [ torch.rand([self.num_candidates,self.num_candidates]) for _ in range(self.popsize)]

    def register_buffer(self, logits, labels):
        r'''

        Args:
            logits (:obj:`torch.Tensor`):
            labels (:obj:`List`):
        '''

        logits = F.softmax(logits.detach(),dim=-1)
        labels = labels.detach()
        if self.probs_buffer is None :
            self.probs_buffer = logits
            self.labels_buffer = labels
        else:
            self.probs_buffer = torch.vstack([self.probs_buffer, logits])
            self.labels_buffer = torch.hstack([self.labels_buffer, labels])

    def process_logits(self, logits: torch.Tensor, **kwargs):

        if self.accumulate: # inherit from nn.Module, only store buffer in training mode.
            self.accumulate_step+=1
            self.register_buffer(logits, kwargs['batch']['label'])

        if hasattr(self, "label_words_ids"): # TODO the content in this "if" is same as super()
            # project
            label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)

            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

            # aggregate
            if label_words_logits.dim()>2:
                label_logits = self.aggregate(label_words_logits)
            else:
                label_logits = label_words_logits
            return label_logits

        else:
            return torch.randn((logits.size(0), self.num_classes), requires_grad=True).to(logits.device)

    def project(self,
                logits: torch.Tensor,
                **kwargs, # TODO
                ) -> torch.Tensor:
        r"""When this verbalizer hasn't perform optimize(), it has no
        ``label_words_ids``, thus will give random predictions, and should
        have no connection to the model to give (miss-leading) grads.

        Args:
            logits (:obj:`torch.Tensor`): The original logits over the vocabulary.

        Returns:
            :obj:`torch.Tensor`: The projected logits of label words.
        """
        label_words_logits = logits[:, self.label_words_ids]
        return label_words_logits


    def optimize(self, model, dataloader, device):
        r"""This is an epoch-level optimize. If used in batch-level like an ordinary
        gradient descend optimizer, the result may not be very satisfying since the accumated
        examples (i.e., the probs_buffer and the labels_buffer) are not enough if the batchsize
        is small.
        """
        # if self.search_id < self.num_searches:
        #     self.label_words_ids = self._find_verbalizer(words_per_label=self.label_word_num_per_class,
        #                                                  num_candidates=self.num_candidates)
        #     self.probs_buffer, self.labels_buffer = None, None
        #     self.search_id += 1
        #     if self.search_id == self.num_searches: # finish optimization
        #         self.accumulate = False
        # else:
        #     logger.info("Verbalizer's max num_searches reached, use the previous label words.")

        self.label_words_ids = self._find_verbalizer(model, dataloader,device)

        self.accumulate = True
        self.probs_buffer, self.labels_buffer = None, None
        
        self._show_verbalizer()


    def _show_verbalizer(self):
        tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in self.label_words_ids]
        # logger.info("Verbalizer is {}".format(tokens))
        print("Verbalizer is {}".format(tokens))
        return tokens

    # 轮盘赌算法( Roulette Wheel Selection )
    def _RWS(self, P):
        m = 0
        r = random.random()
        for i in range(self.popsize):
            m += P[i]
            if r <= m:
                return i

    def evo_search_words(self, model, dataloader,device):
        # mats = [ torch.rand([self.num_candidates,self.num_candidates]) for _ in range(popsize)]
        # for i in range(popsize):
        #     mat = torch.rand([self.num_candidates,self.num_candidates])
        #     mats.append(mat)
        it = 0
        while it < self.max_it:
            # 计算适应值
            scores = []
            for idx,mat in enumerate(self.mats):                
                label_words_ids = self._get_label_words(mat, self.candidate_probs, self.candidate_ids )
                model.verbalizer.label_words_ids = label_words_ids

                alllabels = []
                allpreds = []

                model.verbalizer.accumulate = False
                model.eval()
                with torch.no_grad():
                    for _, inputs in enumerate(dataloader):
                        inputs = inputs.to(device)
                        logits = model(inputs)
                        labels = inputs['label']
                        alllabels.extend(labels.cpu().tolist())
                        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
                
                # best_score, label_words = get_score(model, train_dataloader,dev_dataloader, mat)
                # print('acc:',acc)
                scores.append((acc, idx, label_words_ids))

            scores = sorted(scores,key= lambda x: x[0],reverse=True)  
            # 选择
            scores = scores[:self.popsize]
            # print_scores = [i[0] for i in scores ]
            # print(print_scores)
            indexes = [ i[1] for i in scores]
            self.mats = [self.mats[indexes[i]] for i in range(len(indexes))]

            mats_tmp = copy.deepcopy(self.mats)

            # 适应值
            fitness = [i[0] for i in scores]
            # 轮盘赌概率
            P = [ i/sum(fitness) for i in fitness]


            mats_new = []
            m = 0
            while m < self.popsize:
                p = random.random()
                idx1 = self._RWS(P)
                idx2 = self._RWS(P)

                mat1 = mats_tmp[idx1]
                mat2 = mats_tmp[idx2]
                if p < self.pc:
                    swap_idx = random.randint(0,self.num_candidates-1)
                    # swap
                    swap_tmp =  mat1[swap_idx]
                    mat1[swap_idx] = mat2[swap_idx]
                    mat2[swap_idx] = swap_tmp

                    if p < self.pm:
                        mats_new.append(mat1* mat2)
                        m += 1
                    else:
                        mats_new.append(mat1)
                        m += 1
            self.mats.extend(mats_new)
            it += 1
            # tmp = copy.deepcopy(self.mats)
            # # 交叉变异
            # for mat in tmp:
            #     # mat_left = torch.rand([num_candidates,num_candidates])
            #     # mat_right = torch.rand([num_candidates,num_candidates])
            #     p = random.random()
            #     if p < 0.25:
            #         mat_new = torch.rand([self.num_candidates,self.num_candidates])
            #         self.mats.append(mat*mat_new)
            #         # matBs.append(torch.transpose(mat, 1, 0) )
            #     else:
            #         random_idx = random.randint(0,len(tmp)-1)
            #         mat_swap = tmp[random_idx]
            #         swap_idx = random.randint(0,self.num_candidates-1)
            #         # swap
            #         swap_tmp = mat[swap_idx]
            #         mat[swap_idx] = mat_swap[swap_idx]
            #         mat_swap[swap_idx] = swap_tmp

            #         tmp[random_idx] = mat_swap
            #         self.mats.append(mat)
                # matBs.append(mat_left*mat)
                # matBs.append(mat*mat_right)

        best_label_words_ids = scores[0][2]
        # print(scores[0])
        
        return best_label_words_ids



    def _find_verbalizer(self, model, dataloader,device):

        print("Finding verbalizer ...")
        probs = self.probs_buffer
        labels = self.labels_buffer
        # candidates = self._get_candidates(num_candidates=num_candidates, probs=probs, labels=labels)
        self.candidate_ids, self.candidate_probs = self._get_candidates_probs(num_candidates=self.num_candidates, probs=probs, labels=labels)

        # 进化，选出最好的mat， num_candidates x num_candidates
        
        label_words_ids = self.evo_search_words(model, dataloader, device)

        # label_words_ids =  self._ea_search_words(mat, self.candidate_probs, self.candidate_ids )

        return label_words_ids
        

    def _get_candidates_probs(self,
                        num_candidates: int,
                        probs: torch.Tensor,
                        labels: torch.Tensor,
                        ) -> Dict[str, List[str]]:
        if num_candidates <= 0:
            return [torch.arange(self.vocab_size) for label_id in range(self.num_classes)]

        # log_probs = torch.log(probs+1e-15)
        candidate_probs = []
        candidate_ids = []
        for label_id in range(self.num_classes):
            label_mask = (labels==label_id).to(torch.float).unsqueeze(-1)
            score = torch.sum(probs * label_mask, dim=0)
            # score = F.softmax(score,dim=0)
            # candidate_id = torch.argsort(score, descending=True)[:num_candidates]
            candidates = torch.sort(score, descending=True)
            candidate_ids.append(candidates.indices[:num_candidates])
            candidate_probs.append(candidates.values[:num_candidates])
        candidate_probs = torch.stack(candidate_probs,0)
        candidate_ids = torch.stack(candidate_ids,0)
        return candidate_ids,candidate_probs

    def _get_label_words(self,
                        mat: torch.Tensor,
                        candidate_probs: torch.Tensor,
                        candidate_ids: torch.Tensor):
        # A = F.softmax(candidate_probs)
        A = candidate_probs
        # B = torch.rand([self.num_candidates,self.num_candidates]).cuda()
        B = mat.cuda()
        # B = F.softmax(B)
        C = torch.mm(A,B)
        softmax_C = F.softmax(C,dim=1)
        label_word_ids = torch.argsort(softmax_C, descending=True)[:,:self.label_word_num_per_class]
        final_word_ids = []
        for label_id in range(self.num_classes):
            final_word_ids.append(candidate_ids[label_id][label_word_ids[label_id]])
        final_word_ids = torch.stack(final_word_ids)

        return final_word_ids

    def from_file(self,
                  path: str,
                  choice: Optional[int] = 0 ):
        raise NotImplementedError("This verbalizer is learned and can't be set from file.")
