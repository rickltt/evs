#!/bin/sh
DIR="./datasets"
mkdir $DIR
cd $DIR

rm -rf agnews
wget --content-disposition https://cloud.tsinghua.edu.cn/f/0fb6af2a1e6647b79098/?dl=1
tar -zxvf agnews.tar.gz 
rm -rf agnews.tar.gz

rm -rf dbpedia
wget --content-disposition https://cloud.tsinghua.edu.cn/f/362d3cdaa63b4692bafb/?dl=1
tar -zxvf dbpedia.tar.gz
rm -rf dbpedia.tar.gz

rm -rf imdb
wget --content-disposition https://cloud.tsinghua.edu.cn/f/37bd6cb978d342db87ed/?dl=1
tar -zxvf imdb.tar.gz 
rm -rf imdb.tar.gz

rm -rf amazon
wget --content-disposition https://cloud.tsinghua.edu.cn/f/e00a4c44aaf844cdb6c9/?dl=1
tar -zxvf amazon.tar.gz
mv datasets/amazon/ amazon
rm  -rf ./datasets
rm -rf amazon.tar.gz

cd ..
