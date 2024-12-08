# PolarStar AI Infra

[북극성 서비스 정보 바로가기](https://github.com/KDT-AiVENGERS/PolarStar_Info)

# ❗️ About AiVENGERS Infra Repository

### Infra Repository는 AiVENGERS의 북극성 프로젝트에서 시작되었으며,

### "인공지능 모델을 효과적으로 개발하고 테스트" 하기 위해 만들어졌습니다.

![aiinfra_diagram](https://github.com/KDT-AiVENGERS/AIInfra/assets/60493070/3bdcd842-cbf8-4a31-926d-77b619284f4e)

<p align="center"><i>
PolarStar_AIInfra 전체 다이어그램
</i>
</p>

<br />

Data Team이 빠르게 데이터를 수집하는 동안

Infra 팀은 얻어진 데이터를 바탕으로 바로바로 다양한 모델을 빠르게 실험해볼 수 있도록

실험 인프라를 구축하였습니다. 


정해진 규격 형태로 Configuration과 코드 파일을 적절히 배치하고 
돌리기만 하면 

학습과 평가가 수행되며, 그 결과를 WandB에서 바로 확인해볼 수 있습니다.

# 🧑🏻‍💻 개발 목표 및 방향성

### 1. 규격화된 코드📌

- 코드 작성 방식을 일관화하여, 처음 코드를 읽는 사람이 이해하기 쉽도록 설계합니다.
- 규격화된 코드를 이용해 코드의 재사용성을 높입니다.
- 코드가 규격화되어 있다는 점을 이용해 실험을 시작할 때 자동으로 코드를 생성할 수 있게 설계합니다. (실제로 수행하지는 못함)

### 2. 실험 시 코드 수정 최소화📌

- 모델 실험 시 코드 수정을 최소화하도록 하여 발생할 수 있는 오류의 경우의 수를 최소화합니다.
- 모델 실행 코드와 실험 Configuration을 완전히 분리하여 실험을 용이하게 합니다.

### 3. 실험 결과 자동 Logging📌

- 모델 실험 시 해당 실험이 어떤 Configuration에서 수행되었는지를 자동으로 기록할 수 있게 코드를 구성합니다.
- 실험이 진행되는 동안, 실험이 종료되었을 때 실험 결과 및 현황을 즉각 시각화하여 인사이트를 얻을 수 있도록 코드를 구성합니다.

### 4. 작업 단위 별로 Module 화📌

- 각 작업 단위별로 모듈화하고 코드 간 상호 종속성을 최소화 시키도록 코드를 구성하여 Test 및 Refactoring, 코드의 일부 변형이 용이하도록 합니다.
- 모듈을 마치 레고 조립하듯이 조립하여 활용할 수 있게 구성하여 실험 시 코드를 쉽게 구성할 수 있도록 설계합니다.

### 5. 협업에 용이하도록 모듈 구성📌

- 협업에 용이할 수 있도록 모듈 및 프로젝트 구조를 설계합니다.
- 협업한 결과를 쉽게 비교분석할 수 있도록 코드를 구성합니다.

### 6. 그 외 추가적인 편의 기능 반영 및 확장성 확보📌

- Pytorch의 device Setting 하는 부분 등 부수적인 것은 자동화되어 돌아갈 수 있도록 설계합니다.
- Sweeping 자동화 및 시각화 등의 추가 기능을 반영합니다.


<br />
위와 같은 목표를 달성하기 위해

저희는 아래와 같은 프레임워크 및 라이브러리를 활용하였습니다.

# ⚒️ 활용된 라이브러리 및 선정 이유

## 1. Pytorch Lightning📌

![image](https://github.com/KDT-AiVENGERS/AIInfra/assets/60493070/4dc0e03d-9249-4948-b4e3-fdc6c2981dbf)

Pytorch Lightning은 AI Model 코드를 규격화하고, 이를 활용한 다양한 편의 기능과 Model Lifecycle 관리를 도와주는 라이브러리입니다. 선정 이유는 아래와 같습니다.

- 코드를 규격화 할 수 있다는 1번 목표에 잘 부합하여 선정하였습니다.
- 실험 결과를 Wandb와 연동 시 wandblogger를 지원한다는 점에서 3번 목표, 5번 목표에 잘 부합하여 선정하였습니다.
- Data Module, Model Module, Task Module, Training Module, Inference Module로 분리하기 용이하다는 점에서 4번 목표에 잘 부합하여 선정하였습니다.
- Device를 자동으로 Setting하는 기능 등 다양한 편의 기능을 제공한다는 점에서 6번 목표에 잘 부합하여 선정하였습니다.
<p align="center">
<img width="600" alt="pytorchLightningCode" src="https://github.com/KDT-AiVENGERS/AIInfra/assets/60493070/d17792e6-418c-46c9-b035-6e6047dcecb4">
</p>
<p align="center"><i>
위의 그림과 같이 정해진 이름의 함수를 구현하는 방식을 사용하여 규격화된 코드를 구현할 수 있습니다.
</i> 
 </p>

## 2. Hydra📌

<p align="center">
<img width="600" alt="Hydra" src="https://github.com/KDT-AiVENGERS/AIInfra/assets/60493070/67f9b795-ba76-4bc3-b29f-2a228264e2da">
</p>

Hydra는 실험에 관련된 Configutration 파일을 코드로부터 분리할 수 있고 py파일에서 작업 시에는 다양한 plugin도 활용할 수 있도록 하는 라이브러리입니다. 선정 이유는 아래와 같습니다.

- Configuration 파일을 분리하여 실험 시 코드 수정을 최소화할 수 있다는 점에서 2번 목표에 부합하여 선정하였습니다.
- Configuration 파일로 실험 내역이 자동 저장되도록 설계가 가능하다는 점에서 3번 목표에 부합하고 파일 형태이므로 공유가 용이하다는 점에서 5번 목표에 부합하여 선정하였습니다.
<p align="center">
<img width="678" alt="Configuration file" src="https://github.com/KDT-AiVENGERS/AIInfra/assets/60493070/d93206d5-b5be-4e72-b8c4-67514b9bf7b6">
</p>
<p align="center"><i>
위의 그림과 같이 Config.yaml 파일에서 실험에 관련된 여러 설정을 세팅할 수 있습니다. <br /> 즉 코드를 전혀 만질 필요없이 이 파일만 건드리면 됩니다.
</i>
</p>

## 3. Wandb📌

<p align="center">
<img width="600" alt="wandb" src="https://github.com/KDT-AiVENGERS/AIInfra/assets/60493070/485a40a0-90de-4a93-8c99-366a3679a945">
</p>

Wandb는 실험 결과를 시각화하여 클라우드에 저장하고 대시보드로 쉽게 확인할 수 있는 환경을 제공하며, 협업 시 서로 다른 실험 간의 결과를 쉽게 비교 및 시각화할 수 있는 툴입니다. 더불어 자동 Sweeping 기능도 제공하고 결과도 시각적으로 확인할 수 있습니다.

- 실험 결과가 클라우드에 자동으로 저장되고 시각화도 용이하다는 점에서 3번 목표에 잘 부합하여 선정하였습니다.
- 서로 다른 컴퓨터의 실험 결과를 한 대시보드에서 비교 분석하기 매우 쉬운 환경을 제공하기 때문에 5번 목표에 잘 부합하여 선정하였습니다.
- Sweeping 자동화를 지원하며 이 결과를 시각화까지 해준다는 점에서 6번 목표에 잘 부합하여 선정하였습니다.

<p align="center">
<img width="1725" alt="wandb dashboard" src="https://github.com/user-attachments/assets/6bcf5595-a29c-4259-bdae-399774978198">

<p align="center"><i>
위의 그림과 같이 WandB Dashboard를 이용해 결과를 시각화할 수 있고 <br />이 결과를 팀원과 공유할 수 있습니다.</i>
</p>

## 4. Transformers - HuggingFace📌

<p align="center">
<img width="600" alt="transformers" src="https://github.com/KDT-AiVENGERS/AIInfra/assets/60493070/88bc9608-3a31-4aa2-8e54-de73ddcffd03">
</p>

Transformers 라이브러리는 HuggingFace Repository에 올라와 있는 모델을 쉽게 가져와서 사용할 수 있고 finetuning 등의 작업도 용이하게 할 수 있는 라이브러리입니다.

- 주요한 모델을 쉽게 가져올 수 있고 HuggingFace에 올라와있는 주요 pretrained 모델을 활용하기 위해 선정했습니다.
- Repository를 활용하면 서로 다른 컴퓨터에서의 모델 결과 값을 쉽게 불러올 수 있다는 점에서 5번 목표에 잘 부합하여 선정하였습니다.

<p align="center">
<img width="1554" alt="huggingface repository" src="https://github.com/KDT-AiVENGERS/AIInfra/assets/60493070/694e59bd-8eb3-4024-8e6e-862bb90b961a">
</p>
<p align="center"><i>
위의 그림과 같이 모델 Config와 Weight값을 HuggingFace Repository에 저장하고<br />이것을 코드 한 줄로 쉽게 불러올 수 있습니다.</i>
</p>

# 🐻‍❄️ Infra를 프로젝트에 적용한 방법

## 1. Infra Repository 기반의 코드 작성 (Infra_Init)📌

본격적으로 저희 프로젝트에 적용할 Infra Code를 작성하는 단계입니다.

- 먼저 간단한 MNIST 문제를 CNN으로 푸는 모델을 이용해서 상기된 목표를 달성하는 Infra Code를 작성합니다.
  MNIST 문제는 가장 유명한 예시이고 코드 자체가 널리 알려져 있고 입문 코드이니, 이를 활용하여 초안 Infra Code를 작성하였습니다.

  `infra_init/mnist_baseline.ipynb`
  `infra_init/config.yaml`
  `infra_init/global.yaml`

- 그 다음, Hugging Face Module을 적용해보기 위해 위에서 작성한 코드의 CNN 모델 부분을 Hugging Face ViT 모델로 대체합니다.
  `infra_init/mnist_vit_baseline.ipynb`
- 그 다음, 이 Infra Code를 자연어 처리 문제에 적용할 수 있도록, 기존에 해결했던 자연어 처리 Task 코드를 가져와 일부 리팩토링합니다.
  `infra_init/nlp_baseline.ipynb`

## 2. 프로젝트에 사용될 모델 관련 분석 및 코드 작성 (Model_Analysis)📌

북극성 프로젝트의 서비스 중 공고 추천 서비스의 구현 방법을 고민해보았습니다. 저희가 결정한 방법은 사용자에게 QnA를 통해 얻어낸 데이터를 활용해 가상의 Job Description인 Mock JD를 만든 뒤, 이 Mock JD를 바탕으로 임베딩 벡터를 뽑아내어, 이 Mock JD와 가장 유사한 임베딩 벡터 값을 가진 JD를 찾아 출력해주는 방식으로 작동 방식을 결정하였습니다.

우리는 자연어 임베딩 모델을 결정하기 위해 여러 경우의 수를 고려해보았고 최종적으로는 BERT 계열의 모델을 사용하기로 하였습니다. 그 이유는 다음과 같습니다.

- 이미 많은 데이터를 통해 문장의 맥락과 주요 단어의 의미를 잘 이해하고 있는 모델이 Pretrained Model로서 제공됩니다.
- Transformer의 Encoder 부분을 활용하므로 Model 의 형태가 Embedding Vector를 추출해 내는 것에 특화되어 있습니다.
- 저희가 Target층으로 정한 개발 분야의 Job Description에 대하여 LLM을 Domain Adaptation을 시켜야 하는데, BERT의 경우 적은 데이터만을 이용해서 이러한 목적을 달성하기에 용이합니다.
- 학습 방식이 MLM 방식이므로 labeling 작업이 필요치 않아, 새로운 JD 데이터를 모델에 적용하는데 문제가 없습니다.

정리하면 BERT 모델을 활용하여 문장을 의미를 가지는 벡터로 변환하고, 공고의 문장과 사용자 질의응답의 문장으로부터 변환된 벡터끼리의 유사도를 계산하여 추천 공고를 결정하는 방식을 채택하였으며 이는 Transformer Encoder 의 구조로 훈련된 BERT 모델의 임베딩 레이어와 히든 레이어가 출력으로 내놓을 벡터가 문장의 맥락과 주요 단어의 의미를 잘 내포하고 있을 것이라는 가정을 바탕으로 하며, 기 훈련된 모델을 활용하는 만큼 Domain Adaptation에 대한 Cost가 낮고 라벨링 된 데이터가 없이 이를 적용할 수 있다는 장점이 있다는 점을 이유로 합니다.

BERT 모델은 Hugging face 를 통해 매우 편리하게 활용 가능합니다. Hugging face 라이브러리를 import 한 후, 아래와 같이 tokenizer 와 model 을 불러오기 하여 즉시 사용 가능합니다.

```python
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
```

모델의 fine-tuning 은 필요없지만, 본 프로젝트에서 주로 다룰 문장은 공고와 사용자 질의, 구직 관련 키워드에 한정되어 있으므로, 모델의 성능을 향상시키기 위해서는 domain-adaptation 과정이 필요하다는 의견이 제기되었습니다. Domain-adaptation 을 위해서는 BERT 모델의 훈련 방법으로 알려져 있는 Masked language modeling(MLM) 방식과 Next sentence prediction(NSP) 방식의 훈련이 필요하며, 훈련에 사용하는 데이터는 수집한 공고 문장 데이터를 그대로 활용 가능하므로 별도의 라벨링 작업은 필요하지 않습니다.

- Hugging Face 에서 MLM 용 모델을 불러와, 보유한 데이터로 MLM 학습시키는 코드와 학습한 모델을 활용하여 주어진 문장으로부터 encoded vector 를 추출하는 코드를 작성합니다.
  `model_analysis/Bert_embedding.ipynb`
- BERT 계열의 다양한 모델을 Hugging Face에서 불러오는 코드를 작성합니다.
  `model_analysis/Bert_various.ipynb`
- BERT 모델에서 나온 결과를 바탕으로 Cosine Similarity를 계산하는 코드를 작성합니다.
  `model_analysis/Bert_cosine_cal.ipynb`

## 3. 모델 성능 비교 테스트 전략 설정📌

- 모델의 성능을 평가하기 위해 2가지 지표를 사용합니다.
- 첫 번째 지표는 model 학습과정에서의 loss 값입니다. 이 값은 작을수록 좋은 모델로 판별합니다.
- 첫 번째 지표는 wandb 대시보드에서 시각화된 결과를 이용해 쉽게 비교분석할 수 있습니다.
- 두 번째 지표는 TestSheet Score입니다. 유사도가 높게 측정되어야 하는 예시 Set와 유사도가 낮게 측정되어야 하는 예시 Set를 모델에 입력하여 우리가 예상한 유사도대로 잘 나오는지 판별하는 지표입니다.
- 두 번째 지표는 TestSheet Score를 계산해주는 코드를 작성하여 그 결과를 비교분석할 수 있습니다.

## 4. 최종 실험용 Baseline Model을 작성 (Baseline)📌

- 위에서 분석된 결과를 바탕으로 최종 Baseline Model을 작성합니다.
- Pretrained된 모델을 실제 서비스에 적용되는 모델 구조로 이식할 때 2가지 방법을 고안했습니다.
  - `new_prediction_layer_deletion 방식` 
  - `new_prediction_from_pretrained 방식`

  <br/>
  <br/>
  <details>
  <summary>두 방식의 차이</summary>

  Test / Prediction 과정에서
  ### Layer Deletion Code
  ```python
  pooler_before_outputs = outputs[0][0]
  pooler_outputs = torch.nn.functional.tanh(pooler_before_outputs)
  ```

  -> BERT의 Pooler Layer 삭제 (MLM 과정에서 학습되지 않는 레이어, NSP 과정에서만 학습 - 어차피 문장의 대표 뜻만 알아내면 되고, 파인튜닝에 쓰이지도 않음 - 의미 없는 레이어다! 라는 가정)

  ### From Pretrained Code
  ```python
  pooler_outputs = outputs[1]
  ```

  -> BERT의 Pooler Layer를 Pretrained Layer를 그대로 활용 (Pretrain 된 결과를 활용해 구조를 그대로 유지함)
  
  </details>
  <br/>
  <br/>
  
  
  
  
- 구성된 Baseline Model 코드 파일들의 구성은 아래와 같습니다.
  - `train_from_pretrained.ipynb` 또는 `train_layer_deletion.ipynb` (최종 모델 학습 코드)
  - `config.yaml` (실험 설정)
  - `global.yaml` (Versioning을 위한 파일)
  - `test_from_pretrained.ipynb` 또는 `test_layer_deletion.ipynb` (2번째 지표를 얻어내기 위한 코드)
  - `test_case.csv` (2번째 지표를 판별하는데 사용되는 test case가 들어있는 csv입니다.)

## 5. 실험 계획 Sheet 작성 및 이를 바탕으로 Experiment 배분 (Experiment)📌

- 팀원 각각이 어떤 실험을 진행할지 실험 계획 Sheet를 작성합니다.
- 작성된 계획 Sheet를 바탕으로 실험을 진행합니다.
- 실험은 experiment 폴더 내에 생성된 각 팀원 폴더에서 실험이 진행됩니다.
- baseline 폴더에 있는 모델을 그대로 가져온 뒤, 실험 계획 Sheet에 나온대로 config파일을 수정한 뒤 모델 학습을 수행합니다.

## 6. 실험 결과 바탕으로 모델 선정
<img width="1509" alt="image" src="https://github.com/user-attachments/assets/84681011-6bab-49a2-93b2-4df3c228ceb4">


# 📁Data Processing Code (DataProcessing)

북극성 프로젝트의 기본 로직은 아래와 같습니다.

1. 사용자와 Q&A 를 주고 받으며 사용자에 대한 정보를 수집합니다.
2. 얻어진 Q&A를 바탕으로 가상의 Mock JD를 얻어냅니다.
3. Mock JD를 BERT 모델을 활용하여 Embedding Vector로 변환합니다.
4. 이 Embedding Vector와 Cosine 유사도가 높은 JD 목록을 얻어냅니다.
5. 이 중 하나의 JD를 선택하여 강의 추천을 요청하면 해당 JD를 바탕으로 겹치는 키워드를 이용하는 알고리즘을 이용해 적절한 강의를 추천합니다.

Data Processing 폴더 내에 있는 폴더 중
JD_feature 폴더는 2번 과정, Lec_recommendlogic 폴더는 5번 과정을 구현한 코드입니다.

## JD_feature📌

- QnA 데이터를 바탕으로 가상 JD를 생성하는 코드가 구현되어 있습니다.
- 더불어 TestSheet를 작성하기 위한 코드도 여기에 구현되어 있습니다.
- 즉 Sample JD와 유사도를 비교하는 코드가 구현되어 있습니다.

## LEC_recommend_logic📌

- JD가 입력으로 들어오면 이와 관련도가 높은 강의를 추천해주는 코드가 구현되어 있습니다.
- 키워드가 매칭되는지 확인하는 방법으로 알고리즘을 구현했습니다.

[Data Processing에 대한 자세한 설명을 확인하려면 여기를 눌러주세요.](https://github.com/KDT-AiVENGERS/PolarStar_Data)

# Retrospection📌

- 인공지능 모델 코드는 대체적으로 비슷한 형태를 따른다. 실제로 Kaggle 등을 진행할 때 이전에 사용했던 코드를 복사해서 붙여넣기해서 코드를 변경하는 일이 잦을 정도로 굉장히 그 형식이 유사하다. 즉 코드의 재사용성이 높으므로 이를 모듈화할 수 있을 것이라는 생각을 하였고 그것을 프로젝트에 적용하여 꽤 만족스러운 성과를 가져왔다는 것이 인상 깊었다.
- kaggle 환경에서 사용할 수 있는 버전으로 테스트를 해보지 못했다. 사실 빠르게 모델 baseline을 코드를 생성하고 테스트해 볼 때 GPU가 없다면 Kaggle 환경에 대한 호환성이 필요한데 아직 이 부분을 제작해보지 못한 것이 아쉽다.
- 모듈화가 되었으므로 코드 자동 생성 코드도 만들어보고 싶었으나, 프로젝트 기간 동안에는 완성하지 못했다. 그러나 이는 향후 재사용 가능성이 높은 코드이므로 코드 자동 생성 코드도 구현해보고자 한다.
