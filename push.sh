#!/usr/bin/env bash

echo "提交到develop分支"
time=`date "+%Y-%m-%d-%H"`
echo "提交时间${time}"
git add . --all
git commit -am ${time}
git push origin develop

echo "提交完成"