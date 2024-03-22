set PATH=%PATH%;C:\projects\winutils\hadoop-3.2.1\bin
call conda env remove --name myenv -y
if %errorlevel% neq 0 exit /b %errorlevel%
call conda create --name myenv python=3.10 -y
if %errorlevel% neq 0 exit /b %errorlevel%
call conda install -c conda-forge -n myenv numpy matplotlib xlrd wordcloud pandas==1.5.3 ipykernel -y
if %errorlevel% neq 0 exit /b %errorlevel%
call conda activate myenv
if %errorlevel% neq 0 exit /b %errorlevel%
call pip install nltk  sparknlp nlu johnsnowlabs pyspark==3.2.3
if %errorlevel% neq 0 exit /b %errorlevel%
python assignment3.py
if %errorlevel% neq 0 exit /b %errorlevel%
