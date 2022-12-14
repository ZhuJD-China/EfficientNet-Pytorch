# EfficientNet-Pytorch

Training your own dataset on EfficientNet by Pytorch and exporting onxx format to test ! ! !

## Step 1：Prepare your own classification dataset

---

Then the data directory should looks like:   

```
-dataset\
    -model\
    -train\
        -1\
        -2\
        ...
    -test\
        -1\
        -2\
        ...
```

## Step 2: train and test

(1)I have choose to download the pre-trained model int the wieghts dic.

(2)Change some settings to match your dataset.

(3)You can get the final results and the best model on ```dataset/model/```.

## Step 3: export onnx format and test

(1) python export_onnx.py (to choose your onw file path!!!)

key code:

```python
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,  # 是否执行常量折叠优化
                    input_names=["input"], # 输入名
                    output_names=["output"])   # 输出名
```

(2) python test_onnx.py to test your onnx model

```python
python .\test_single_img.py
inputs: torch.Size([1, 3, 300, 300])
tensor([[-0.9127, -0.6780, -0.6101, -0.1727, -1.2774,  0.1447,  0.5889,  0.2055,
          9.4926, -1.2727, -1.2145, -1.9575, -0.9537, -1.8501, -0.6297,  1.0857,
          0.4583]], device='cuda:0', grad_fn=<AddmmBackward0>)
8
{'1': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5, '15': 6, '16': 7, '17': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16}
{0: '1', 1: '10', 2: '11', 3: '12', 4: '13', 5: '14', 6: '15', 7: '16', 8: '17', 9: '2', 10: '3', 11: '4', 12: '5', 13: '6', 14: '7', 15: '8', 16: '9'}
<class 'dict'>
./OxFlower17/test/17/1348.jpg
prediction_label: 17
```

(3) python pth_to_wts.py to get model weights params. the xxxxx.wts can import to TensorRT engine to accelerate reasoning !!!


## TensorRT engine inference!!
#### [EfficientNet-TensorRT]: https://github.com/ZhuJD-China/EfficientNet-TensorRT
