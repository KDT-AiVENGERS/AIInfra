echo "Copying user nnp dictionary"
! cp ./user-nnp.csv /tmp/user-nnp.csv

echo "Change Directory to /tmp"
cd /tmp

echo "Installing konlpy....."
! pip3 install konlpy
echo "Done"


echo "Installing mecab-0.996-ko-0.9.2.tar.gz....."

echo "Downloading mecab-0.996-ko-0.9.2.tar.gz......."
echo "from https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz"
! wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz -nc
echo "Done"

echo "Unpacking mecab-0.996-ko-0.9.2.tar.gz......."
! tar xvfz mecab-0.996-ko-0.9.2.tar.gz > /dev/null 2>&1
echo "Done"

echo "Change Directory to mecab-0.996-ko-0.9.2......."
cd mecab-0.996-ko-0.9.2/
echo "Done"

echo "installing mecab-0.996-ko-0.9.2.tar.gz........"
echo 'configure'
! ./configure > /dev/null 2>&1
echo 'make'
! make > /dev/null 2>&1
echo 'make check'
! make check > /dev/null 2>&1
echo 'make install'
! make install > /dev/null 2>&1

echo 'ldconfig'
! ldconfig > /dev/null 2>&1
echo "Done"

echo "Change Directory to current folder"
cd ../

echo "Downloading mecab-ko-dic-2.1.1-20180720.tar.gz......."
echo "from https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz"
! wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz -nc
echo "Done"

echo "Unpacking  mecab-ko-dic-2.1.1-20180720.tar.gz......."
! tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz > /dev/null 2>&1
echo "Done"

echo "Change Directory to mecab-ko-dic-2.1.1-20180720"
cd mecab-ko-dic-2.1.1-20180720/
echo "Done"

echo "installing........"
echo 'configure'
! ./configure > /dev/null 2>&1
echo 'make'
! make > /dev/null 2>&1
echo 'make install'
! make install > /dev/null 2>&1

echo 'bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/v0.6.0/scripts/mecab.sh)'
! bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/v0.6.0/scripts/mecab.sh)  > /dev/null 2>&1
echo "Done"

echo "Install mecab-python"
! pip install mecab-python > /dev/null 2>&1

cp ../user-nnp.csv ./user-nnp.csv

echo 'make clean'
! make clean > /dev/null 2>&1
echo 'make install'
! make install > /dev/null 2>&1

echo "Change Directory to current folder"
! cd ../
echo "Done"

! rm mecab-0.996-ko-0.9.2.tar.gz
echo "Delete mecab-0.996-ko-0.9.2.tar.gz"
! rm -r mecab-0.996-ko-0.9.2
echo "Delete mecab-0.996-ko-0.9.2"
! rm mecab-ko-dic-2.1.1-20180720.tar.gz
echo "Delete mecab-ko-dic-2.1.1-20180720.tar.gz"
! rm -r mecab-ko-dic-2.1.1-20180720
echo "Delete mecab-ko-dic-2.1.1-20180720"
! rm user-nnp.csv
echo "Delete user-nnp.csv"

echo "Successfully Installed"

echo "Now you can use Mecab"
echo "from konlpy.tag import Mecab"
echo "mecab = Mecab()"

echo "사용자 사전 추가 방법 : https://bit.ly/3k0ZH53"

echo "NameError: name 'Tagger' is not defined 오류 발생 시 런타임을 재실행 해주세요"
echo "Reference: https://github.com/SOMJANG/Mecab-ko-for-Google-Colab"