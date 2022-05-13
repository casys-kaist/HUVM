#!/bin/bash
mkdir -p ../dataset/graph
cd ../dataset/graph
wget -c https://rapidsai-data.s3.us-east-2.amazonaws.com/cugraph/benchmark/benchmark_csv_data.tgz
tar -xvf benchmark_csv_data.tgz
mv undirected/soc-twitter-2010.csv .
wget -c https://nrvis.com/download/data/soc/soc-sinaweibo.zip
unzip -o soc-sinaweibo.zip
wget -c https://nrvis.com/download/data/massive/web-ClueWeb09-50m.zip
unzip -o web-ClueWeb09-50m.zip
wget -c https://nrvis.com/download/data/massive/web-uk-2005-all.zip
unzip -o web-uk-2005-all.zip
wget -c https://nrvis.com/download/data/massive/web-cc12-PayLevelDomain.zip
unzip -o web-cc12-PayLevelDomain.zip
wget -c https://nrvis.com/download/data/massive/web-wikipedia_link_en13-all.zip
unzip -o web-wikipedia_link_en13-all.zip
