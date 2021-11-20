#!/bin/zsh
DATE=$(date "+%Y-%m-%d")

#source /Users/choij/workspace/Finance/setup.sh
source /root/miniconda3/bin/activate
conda activate finance
python /root/workspace/SystemTrading/SQLTools/DBUpdater.py --market KRX --update \
	>& /root/workspace/SystemTrading/SQLTools/logs/updateDaily_$DATE.log
python /root/workspace/SystemTrading/SQLTools/DBUpdater.py --market ETF --update \
	>& /root/workspace/SystemTrading/SQLTools/logs/updateDaily_$DATE.log
