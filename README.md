# RethNet

Object-by-Object Learning for Detecting Facial Skin Problems

`tmp`

```
ConvLSTM 적용 시, Patch Pixels 를 Time series 로 활용하나?
맞다면, 그로써 얻어지는 이점은 무엇인가?

ConvLSTM의 output으로 return_sequences옵션에 따라 값이 달라지는데 이에 따른 의미를 파악하고
어떤 방법으로 사용할지 결정해야 한다.

ConvLSTM 의 입력으로 [B, Pixels, Patch_size*2, Channels], 계산은 pixels 간의 순차 특성을 계산함.
본문에서는 SE와의 조합으로 Local context와 Global Context를 효과적으로 Encoding 할 수 있다고 밝힘.
하지만 LSTM은 단방향 Encoding으로 Local context를 일부 추출할 수 있을지라도, 
전체 Context를 읽는 것은 힘들다 사료됨.
따라서 Self-Attention으로 대체하는 것이 효과적일 것이라 생각됨.


Paper 에서는 주변 Patches의 lower-level contextual information,  
spatial correlation을 추출함을 명시함


```

* **[`tmp`](https://127.0.0.1)**:
    tmp
