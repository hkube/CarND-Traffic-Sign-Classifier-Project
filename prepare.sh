#!/bin/bash

wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
mkdir traffic-signs-data
cd traffic-signs-data
unzip -x ../traffic-signs-data.zip
cd ..
source activate carnd-term1
git config --global --add alias.tree "log --decorate --oneline"
git config --global --add alias.st status
git config --global --add alias.ci commit
git config --global --add alias.co checkout
git config --global --add user.email "harald.kube@gmx.de"
git config --global --add user.name  "Harald Kube"
