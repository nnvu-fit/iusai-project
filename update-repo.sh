## curl source code from github
wget 'https://github.com/nnvu-fit/iusai-project/archive/refs/heads/main.zip' -O main.zip
unzip main.zip
mv -f iusai-project-main/* .
rm -rf iusai-project-main main.zip