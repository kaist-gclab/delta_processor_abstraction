# Starlab 7년차 <br />
## Mesh Abstraction <br />
### Set Environment 🚀
```
conda env create -f environment.yml
conda activate mcnenv
```

### Download Princeton Segmentation Benchmark Dataset for Model 📦️ <br />
[Download Dataset](https://drive.google.com/file/d/1T09piyXOaEpxwgwcyZRJIOnLx-NpZ8yr/view?usp=sharing) <br />
이 repository의 가장 상위 폴더에 datasets가 위치하게 압축 해제 해두시면 됩니다.

### Test Mesh Segmentation Model using PSB 🧪
```
bash ./run_test3.sh
```

### Explanation of Direct Functions 💡

**1. mesh_abstraction.py** <br />
+ process_seg.py로 계산한 pseg와 mesh를 통해 abstraction을 진행하고, abstraction volume을 %로 계산한다.<br />
run_test3.sh에 직접적으로 사용되는 테스트 모듈이다.

**2. process_seg.py** <br />
+ abstraction을 구할 face label을 datasets/prince_abs_1000 파일에 저장해준다

**3. simp_visualize.py (DEBUG)** <br />
+ simplified 된 mesh의 visualization 결과를 보여줍니다.<br />
하나의 모델 당 여러 사람이 분류한 segmentation gt가 존재합니다.<br />
이중 균일한 gt를 사용해야하기 때문에 확인용으로 만들어두었습니다.

+ 모든 Segmentation 확인하기<br />
L43-47을 돌리면 각 class의 mesh별 모든 segmentation을 볼 수 있습니다.<br />
Segmentation이 21개 이하인 경우만 볼 수 있고 더 많은 경우에는 스킵하도록 설정되어있습니다.

+ List를 기반으로 균일한 Segmentation 확인하기<br />
L38-42를 돌리면 각 class에서 선택한 하나의 균일한 segmentation을 볼 수 있습니다.<br />
simp_seg_label 파일에 txt파일로 각 클래스별 segmentation division이 저장되어 있습니다.<br />
해당되는 클래스의 list를 복사하여 dictionary부분에 복사하면 됩니다.

### Explanation of Indirect Functions ✨
1. util.py: 파일 불러오기 및 저장 관련 함수들
2. volume_util.py: part mesh, obb, aabb 계산 및 abstraction area 계산해주는 함수들
3. visualize.py: mesh visualization 관련 함수들
