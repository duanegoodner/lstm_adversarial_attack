| Layer # | Description        | Input Shape            | Parameters          | Ouput Shape            |   Activation  |
|---------| :------------------| :----------------------| :-------------------|------------------------|:--------------|
| 1       | Bidirectional LSTM | (b, s, m)              | h<sub>LSTM</sub>    | (b, 2h<sub>LSTM</sub>) | ReLU or Tanh  |
| 2       | Dropoout           | (b, 2h<sub>LSTM</sub>) | P<sub>dropout</sub> | (b, 2h<sub>LSTM</sub>) | -             |
| 3       | Fully Connected    | (b, 2h<sub>LSTM</sub>) | h<sub>FC</sub>      | (b, h<sub>FC1</sub>)   | ReLU or Tanh  |
| 4       | Output             | (b, h<sub>FC1</sub>)   | n<sub>out</sub> = 2 | (b, 2)                 | Softmax       |