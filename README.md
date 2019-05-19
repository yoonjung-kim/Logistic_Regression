# Conversion 확률 예측을 위한 Logistic Regression

## 실행 환경

본 프로그램은 Ubuntu 환경 하에서 Python 3를 사용하여 작성 및 구동 되었습니다. 서버 구축을 위해 Flask 라이브러리가 필요합니다.


## 데이터 다운로드

이 repository의 data 디렉토리에 저장된 `data5.txt`는 본래 데이터 파일 중 첫 100,000개의 데이터 포인트만을 포함하고 있습니다. 전체 데이터로 학습하려면 아래의 명령어를 실행하여 Criteo Conversion Logs Dataset의 데이터 파일(`data.txt`)을 다운로드 해야합니다. 
```
wget http://labs.criteo.com/wp-content/uploads/2014/07/criteo_conversion_logs.tar.gz
tar -C data/ -xvzf criteo_conversion_logs.tar.gz data.txt
```


## 모델 학습

아래와 같이 `logistic_regression.py`를 실행하여 모델을 학습하고 테스트합니다. 데이터 파일의 위치와 학습 모델을 저장할 위치를 프로그램 실행 시 지정해야 합니다.

```
python logistic_regression.py <data path> <model save path>
```
예시: 
```
python logistic_regression.py ./data/data.txt ./model/model.dat
```
위의 예시는 data 디렉토리의 data.txt을 사용하여 모델을 학습하며 그 모델을 model 디렉토리에 model.dat으로 저장합니다.


## 서버 실행

Flask를 사용하여 모델 서버를 구동합니다. `server.py` 실행 시 저장된 모델의 위치를 지정해야 합니다. 
```
python server.py <model path>
```
예시: 
```
python server.py ./model/model.dat
```
위의 예시는 model 디렉토리에 저장된 model.dat을 사용하는 모델서버를 실행합니다.


Python 파일 실행 후 **http://localhost:8080** 에 접속하여 `data.txt`파일의 임의의 라인을 브러우저에서 입력, conversion 확률을 출력합니다. 서버 실행 중 모델 파일이 변경된 경우, 새로운 모델을 이용하여 확률값이 계산됩니다. **http://localhost:8080/update/model** 으로 접속 시, 모델을 다시 로드합니다.


## 저장된 모델

`model` 디렉토리에는 `model.dat`와 `model5.dat` 파일이 저장되어 있습니다. 각각의 모델 파일은 `data.txt`(전체 데이터)와 `data5.txt`(100,000개의 데이터 포인트만 포함)를 이용하여 학습되었습니다.