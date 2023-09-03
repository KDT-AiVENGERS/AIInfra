# PolarStar AI Infra

[북극성 통합 repository 바로가기](https://github.com/KDT-AiVENGERS/.github/tree/develop/profile/polarstar)

## ❗️ About AiVENGERS Infra Repository

### Infra Repository는 AiVENGERS의 북극성 프로젝트에서 시작되었으며,

### "인공지능 모델을 효과적으로 개발하고 테스트" 하기 위해 만들어졌습니다.

<br />

백엔드 개발에서 서버실 → IDC → 클라우드 서비스로 넘어오면서

개발자들은 부수적인 것을 관리하는 시간을 줄이고

오로지, 개발과 서비스에 집중할 수 있는 환경을 만들어왔습니다.

그리고 이것은 큰 생산성과 효율성의 증대로 이어졌습니다.

<br />

인공지능 개발에서도, 인공지능 개발자가 서비스에 적합한 모델을 선정하고

이를 비교 평가 및 분석하고 인사이트를 얻는데에 집중할 수 있는 개발 환경 및 인프라가 갖추어져 있으면

참 좋을 것 같다고 생각하였습니다.

<br />

코드에서 발생하는 부수적인 문제에 신경쓰는 시간을 최소화하고

생각을 즉각적으로 코드로 구현하고 즉각적으로 확인 및 분석하에

빠르고 효과적인 인공지능 개발 환경을 만들어 낼 수 있도록 인프라를 구축하였습니다.

이 레포지토리에서는 위와 같은 실험 Infra 코드와 이를 북극성 프로젝트에 적용한 코드를 담고 있습니다.

Infra Repository는 아래와 같은 개발환경을 목표로 개발되었습니다.

## 🧑🏻‍💻 개발 목표 및 방향성

### 1. 규격화된 코드📌

- 코드 작성 방식을 일관화하여, 처음 코드를 읽는 사람이 이해하기 쉽도록 설계합니다.
- 규격화된 코드를 이용해 코드의 재사용성을 높입니다.
- 코드가 규격화되어 있다는 점을 이용해 실험을 시작할 때 자동으로 코드를 생성할 수 있게 설계합니다.

### 2. 실험 시 코드 수정 최소화📌

- 모델 실험 시 코드 수정을 최소화하도록 하여 발생할 수 있는 오류의 경우의 수를 최소화합니다.
- 모델 실행 코드와 실험 Configuration을 완전히 분리하여 실험을 용이하게 합니다.

### 3. 실험 결과 자동 Logging

- 모델 실험 시 해당 실험이 어떤 Configuration에서 수행되었는지를 자동으로 기록할 수 있게 코드를 구성합니다.
- 실험이 진행되는 동안, 실험이 종료되었을 때 실험 결과 및 현황을 즉각 시각화하여 인사이트를 얻을 수 있도록 코드를 구성합니다.

### 4. 작업 단위 별로 Module 화

- 각 작업 단위별로 모듈화하고 코드 간 상호 종속성을 최소화 시키도록 코드를 구성하여 Test 및 Refactoring, 코드의 일부 변형이 용이하도록 합니다.
- 여건이 허락한다면 TDD를 적용하여 각 모듈별로 독립적인 TestCase를 통과하는지 여부를 테스트하여 코드 테스트를 용이하게 합니다. (추후 계획 중)
- 모듈을 마치 레고 조립하듯이 조립하여 활용할 수 있게 구성하여 실험 시 코드를 쉽게 구성할 수 있도록 설계합니다.

### 5. 협업에 용이하도록 모듈 구성

- 협업에 용이할 수 있도록 모듈 및 프로젝트 구조를 설계합니다.
- 협업한 결과를 쉽게 비교분석할 수 있도록 코드를 구성합니다.

### 6. 그 외 추가적인 편의 기능 반영 및 확장성 확보

- Pytorch의 device Setting 하는 부분 등 부수적인 것은 자동화되어 돌아갈 수 있도록 설계합니다.
- Sweeping 자동화 및 시각화 등의 추가 기능을 반영합니다.

### 7. Baseline 자동 생성 코드 (추후 계획 중)

- 규격화된 코드, TDD 기반의 모듈화된 코드인 점을 활용하여 Baseline을 자동 생성하고, 모델 수정 시 TDD의 TestCase를 통과하는지 여부를 통해 코드를 쉽게 테스트하고 변형 및 재사용 할 수 있게끔 코드를 구성합니다.

### 8. 함수형 프로그래밍적 설계 요소 반영 (추후 리팩토링 계획 중)

- 비순수함수의 비중을 최대한 줄이고, 순수함수의 비중을 높여서 코드 관리를 용이하게끔 합니다.
- 비순수함수 내에서 최대한 순수함수 적인 요소는 함수를 분리하여 설계합니다.

<br />
위와 같은 목표를 달성하기 위해

저희는 아래와 같은 프레임워크 및 라이브러리를 활용하였습니다.

## ⚒️ 활용된 라이브러리 및 선정 이유

### 1. Pytorch Lightning

Pytorch Lightning은 AI Model 코드를 규격화하고, 이를 활용한 다양한 편의 기능과 Model Lifecycle 관리를 도와주는 라이브러리입니다. 선정 이유는 아래와 같습니다.

- 코드를 규격화 할 수 있다는 1번 목표, 7번 목표에 잘 부합하여 선정하였습니다.
- 실험 결과를 Wandb와 연동 시 wandblogger를 지원한다는 점에서 3번 목표, 5번 목표에 잘 부합하여 선정하였습니다.
- Data Module, Model Module, Task Module, Training Module, Inference Module로 분리하기 용이하다는 점에서 4번 목표에 잘 부합하여 선정하였습니다.
- Device를 자동으로 Setting하는 기능 등 다양한 편의 기능을 제공한다는 점에서 6번 목표에 잘 부합하여 선정하였습니다.

### 2. Hydra

Hydra는 실험에 관련된 Configutration 파일을 코드로부터 분리할 수 있고 py파일에서 작업 시에는 다양한 plugin도 활용할 수 있도록 하는 라이브러리입니다. 선정 이유는 아래와 같습니다.

- Configuration 파일을 분리하여 실험 시 코드 수정을 최소화할 수 있다는 점에서 2번 목표에 부합하여 선정하였습니다.
- Configuration 파일로 실험 내역이 자동 저장되도록 설계가 가능하다는 점에서 3번 목표에 부합하고 파일 형태이므로 공유가 용이하다는 점에서 5번 목표에 부합하여 선정하였습니다.

### 3. Wandb

Wandb는 실험 결과를 시각화하여 클라우드에 저장하고 대시보드로 쉽게 확인할 수 있는 환경을 제공하며, 협업 시 서로 다른 실험 간의 결과를 쉽게 비교 및 시각화할 수 있는 툴입니다. 더불어 자동 Sweeping 기능도 제공하고 결과도 시각적으로 확인할 수 있습니다.

- 실험 결과가 클라우드에 자동으로 저장되고 시각화도 용이하다는 점에서 3번 목표에 잘 부합하여 선정하였습니다.
- 서로 다른 컴퓨터의 실험 결과를 한 대시보드에서 비교 분석하기 매우 쉬운 환경을 제공하기 때문에 5번 목표에 잘 부합하여 선정하였습니다.
- Sweeping 자동화를 지원하며 이 결과를 시각화까지 해준다는 점에서 6번 목표에 잘 부합하여 선정하였습니다.

### 4. Transformers - HuggingFace

Transformers 라이브러리는 HuggingFace Repository에 올라와 있는 모델을 쉽게 가져와서 사용할 수 있고 finetuning 등의 작업도 용이하게 할 수 있는 라이브러리입니다.

- 주요한 모델을 쉽게 가져올 수 있고 HuggingFace에 올라와있는 주요 pretrained 모델을 활용하기 위해 선정했습니다.
- Repository를 활용하면 서로 다른 컴퓨터에서의 모델 결과 값을 쉽게 불러올 수 있다는 점에서 5번 목표에 잘 부합하여 선정하였습니다.

Infra Model의 구체적인 구조와 사용 방법은 Wiki 를 참고해주세요.

## 🐻‍❄️ Infra를 북극성 Project에 적용

Infra 저장소는 북극성 프로젝트를 위한 모델을 개발하고 테스트하기 위한 목적으로 만들어졌습니다. 효율적인 인공지능 모델 개발을 위해 개발 코드를 규격화하였으며, 팀원들이 각각 새로운 모델이나 학습 방식을 도입하였을 경우 그 결과를 한 데 모아 비교할 수 있도록 시각화하였습니다.

저희 AiVENGERS Team은 이렇게 구현한 Infra를 바탕으로 북극성 프로젝트에 사용될 인공지능 모델을 실험하고 튜닝하고 인사이트를 얻어 최적화된 최종 모델을 얻어내는 과정에서 적극적으로 활용했습니다.

Infra를 북극성 Project에 적용한 과정은 아래와 같습니다.

### 1. Infra Repository 기반의 코드 작성 (Infra_Init)

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

### 2. 프로젝트에 사용될 모델 관련 분석 및 코드 작성 (Model_Analysis)

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

### 3. 모델 성능 비교 테스트 전략 설정

- 모델의 성능을 평가하기 위해 2가지 지표를 사용합니다.
- 첫 번째 지표는 model 학습과정에서의 loss 값입니다. 이 값은 작을수록 좋은 모델로 판별합니다.
- 첫 번째 지표는 wandb 대시보드에서 시각화된 결과를 이용해 쉽게 비교분석할 수 있습니다.
- 두 번째 지표는 TestSheet Score입니다. 유사도가 높게 측정되어야 하는 예시 Set와 유사도가 낮게 측정되어야 하는 예시 Set를 모델에 입력하여 우리가 예상한 유사도대로 잘 나오는지 판별하는 지표입니다.
- 두 번째 지표는 TestSheet Score를 계산해주는 코드를 작성하여 그 결과를 비교분석할 수 있습니다.

### 4. 최종 실험용 Baseline Model을 작성 (Baseline)

- 위에서 분석된 결과를 바탕으로 최종 Baseline Model을 작성합니다.
- Pretrained된 모델을 실제 서비스에 적용되는 모델 구조로 이식할 때 2가지 방법을 고안했습니다.
  `new_prediction_layer_deletion 방식`
  `new_prediction_from_pretrained 방식`
  > 자세한 설명 보기 (접기 이용, 아래에 상세 설명)
- 구성된 Baseline Model 코드 파일들의 구성은 아래와 같습니다.
  `train_from_pretrained.ipynb` 또는 `train_layer_deletion.ipynb` (최종 모델 학습 코드)
  `config.yaml` (실험 설정)
  `global.yaml` (Versioning을 위한 파일)
  `test_from_pretrained.ipynb` 또는 `test_layer_deletion.ipynb` (2번째 지표를 얻어내기 위한 코드)
  `test_case.csv` (2번째 지표를 판별하는데 사용되는 test case가 들어있는 csv입니다.)

### 5. 실험 계획 Sheet 작성 및 이를 바탕으로 Experiment 배분 (Experiment)

- 팀원 각각이 어떤 실험을 진행할지 실험 계획 Sheet를 작성합니다.
- 작성된 계획 Sheet를 바탕으로 실험을 진행합니다.
- 실험은 experiment 폴더 내에 생성된 각 팀원 폴더에서 실험이 진행됩니다.
- baseline 폴더에 있는 모델을 그대로 가져온 뒤, 실험 계획 Sheet에 나온대로 config파일을 수정한 뒤 모델 학습을 수행합니다.

## JDfeature

## sampleJD

실험을 진행한 모델의 성능을 비교해 보기 위해 미리 분류 해놓은 sample data를 기반으로 cosine 유사도를 계산한 csv파일을 생성하고 각 모델의 loss값을 계산하여 best모델을 선정하였습니다.

### mockJD

질문을 통해 사용자에게 입력받은 내용을 기반으로 가상의 JD파일을 만들고 이 가상의 JD와 코사인 유사도를 통해 공고를 추천 해줍니다.

- 알고리즘 폴더
  [https://github.com/KDT-AiVENGERS/AIInfra/tree/develop/LEC_recommendlogic](https://github.com/KDT-AiVENGERS/AIInfra/tree/develop/LEC_recommendlogic)
