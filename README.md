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

(3) python pth_to_wts.py to get model weights params. the xxxxx.wts can import to TensorRT engine to accelerate reasoning !!!
