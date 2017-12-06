# Your Name <span style="color:red">(趙賀笙 F128843741)</span>

# Project 3: Scene recognition with bag of words

## Overview
這次作業是要用兩種不同的Feature擷取和兩種Classifier來實作圖片辨識，分別是以下三種

1. Tiny images feature detection with nearest neighbor classifier
2. Bag of SIFT feature detection with nearest neighbor classifier
3. Bag of SIFT feature detection with linear SVM classifier

其中第二種的Features擷取是利用bag of words來實作，也就是先從所有的訓練圖片中取出SIFT特徵，再利用K-Means將所有的features分成K群，最後這K群的中心即是一大群的特徵，利用這些特徵在每張圖出現的比率來預測圖片，也可以想像這些特徵即是一群不同類別的物品，再不一樣的場景這些物品出現的次數會不一樣，就可以分辨出場景。


## Implementation
### Features Extraction
1. **Tiny images**(get\_tiny\_images.py)

	將所有圖像縮小到 16x16，重新調整矩陣變成1x256放入回傳的list，等於利用縮小後的像素當做 256 個特徵。
	
	```python
	    tiny_images = []
	    for img_path in image_paths:
	        img = Image.open(img_path)
	        img_resized = np.asarray(img.resize((16, 16), Image.ANTIALIAS)).reshape(1,-1)
	        tiny_images.extend(img_resized)
	    tiny_images = np.asarray(tiny_images)
	    
	```	    	
	
2. **Bag of Word with SIFT features**(get\_bags\_of\_sifts.py)
	
	從每張圖片提取SIFT特徵，並利用 K-Means 分群法來建造一個字典，將features分成不同類別。接著取要預測的圖片的SIFT特徵，將這些feature根據前面建的字典，找到最近的群來分類為不同的word,將這些feature在字典裡面的分佈狀況作為後面分類的依據，進而於測出圖片類別。
	
	```python
		image_feats=[]
	    vocab=pickle.load(open('vocab.pkl', 'rb'))
	
	    for image_path in image_paths:
	        img = np.asarray(Image.open(image_path),dtype='float32')
	        frames, descriptors = dsift(img, step=[5,5], fast=True)
	        distance_matrix = distance.cdist(descriptors,vocab,'euclidean')
	        feature_idx = np.argmin(distance_matrix,axis=1)
	        unique, counts = np.unique(feature_idx, return_counts=True)
	        counter = dict(zip(unique, counts))
	
	        histogram = np.zeros(vocab.shape[0])
	        for idx, count in counter.items():
	            histogram[idx] = count
	        histogram = histogram/histogram.sum()
	
	        image_feats.append(histogram)
	        print(image_path)
	    image_feats = np.asarray(image_feats)
	    
	```

### Classifier
1. **Nearest-Neighbor**(nearest\_neighbor\_classify.py)
	
	把要被預測的圖片的特徵和所有的訓練圖片的特徵比較距離，以最近也就是最相似的圖片的Label作為預測結果。
	
	```python
	    distance_mtx = distance.cdist(test_image_feats, train_image_feats)
	    nn_index = np.argmin(distance_mtx, axis=1)
	    test_predicts = [train_labels[i] for i in nn_index]
	    
	```	

2. **Linear SVM classifier**(svm\_classify.py)
	
	用一個 1-vs-all SVM 來在SIFT特徵上進行分類，也就是利用不同的多項式來逼近每個類別最佳的分類方法，了解了背後的運作原理後試著調整參數，許多參數我都用原本的預設值，改了C的值還有max_iter，找出比較好的結果，不過發現上限大概是66%。
	
	```
		svm = LinearSVC(C= 10, class_weight=None, dual=True, fit_intercept=True,
	                    intercept_scaling=1, loss='squared_hinge', max_iter= 500,
	                    multi_class='ovr', penalty='l2', random_state=0, tol= 0.00005,
	                    verbose=0)
	    svm.fit(train_image_feats, train_labels)
	    pred_label = svm.predict(test_image_feats)  
	    
	```

## Installation

- Install cyvlfeat for fetching sift features


## Visualization
<img src="tiny-NN.png">

tiny_images-NN Accuracy (mean of diagonal of confusion matrix) is 0.203

<img src="bag-NN.png">

BOV-NN Accuracy (mean of diagonal of confusion matrix) is 0.55

<img src="bag-SVM.png">

BOV-SVM Accuracy (mean of diagonal of confusion matrix) is 0.65 


| Category name | Sample training images | Sample true positives | False positives with true label | False negatives with wrong predicted label |
| :-----------: | :--------------------: | :-------------------: | :-----------------------------: | :----------------------------------------: |
| Kitchen | ![](thumbnails/Kitchen_train_image_0146.jpg) | ![](thumbnails/Kitchen_TP_image_0203.jpg) | ![](thumbnails/Kitchen_FP_image_0024.jpg) | ![](thumbnails/Kitchen_FN_image_0149.jpg) |
| Store | ![](thumbnails/Store_train_image_0191.jpg) | ![](thumbnails/Store_TP_image_0254.jpg) | ![](thumbnails/Store_FP_image_0250.jpg) | ![](thumbnails/Store_FN_image_0297.jpg) |
| Bedroom | ![](thumbnails/Bedroom_train_image_0146.jpg) | ![](thumbnails/Bedroom_TP_image_0175.jpg) | ![](thumbnails/Bedroom_FP_image_0008.jpg) | ![](thumbnails/Bedroom_FN_image_0207.jpg) |
| LivingRoom | ![](thumbnails/LivingRoom_train_image_0185.jpg) | ![](thumbnails/LivingRoom_TP_image_0240.jpg) | ![](thumbnails/LivingRoom_FP_image_0047.jpg) | ![](thumbnails/LivingRoom_FN_image_0096.jpg) |
| Office | ![](thumbnails/Office_train_image_0152.jpg) | ![](thumbnails/Office_TP_image_0011.jpg) | ![](thumbnails/Office_FP_image_0356.jpg) | ![](thumbnails/Office_FN_image_0062.jpg) |
| Industrial | ![](thumbnails/Industrial_train_image_0191.jpg) | ![](thumbnails/Industrial_TP_image_0256.jpg) | ![](thumbnails/Industrial_FP_image_0227.jpg) | ![](thumbnails/Industrial_FN_image_0108.jpg) |
| Suburb | ![](thumbnails/Suburb_train_image_0191.jpg) | ![](thumbnails/Suburb_TP_image_0128.jpg) | ![](thumbnails/Suburb_FP_image_0180.jpg) | ![](thumbnails/Suburb_FN_image_0103.jpg) |
| InsideCity | ![](thumbnails/InsideCity_train_image_0152.jpg) | ![](thumbnails/InsideCity_TP_image_0040.jpg) | ![](thumbnails/InsideCity_FP_image_0251.jpg) | ![](thumbnails/InsideCity_FN_image_0054.jpg) |
| TallBuilding | ![](thumbnails/TallBuilding_train_image_0152.jpg) | ![](thumbnails/TallBuilding_TP_image_0053.jpg) | ![](thumbnails/TallBuilding_FP_image_0047.jpg) | ![](thumbnails/TallBuilding_FN_image_0292.jpg) |
| Street | ![](thumbnails/Street_train_image_0152.jpg) | ![](thumbnails/Street_TP_image_0133.jpg) | ![](thumbnails/Street_FP_image_0036.jpg) | ![](thumbnails/Street_FN_image_0080.jpg) |
| Highway | ![](thumbnails/Highway_train_image_0152.jpg) | ![](thumbnails/Highway_TP_image_0104.jpg) | ![](thumbnails/Highway_FP_image_0030.jpg) | ![](thumbnails/Highway_FN_image_0257.jpg) |
| OpenCountry | ![](thumbnails/OpenCountry_train_image_0146.jpg) | ![](thumbnails/OpenCountry_TP_image_0093.jpg) | ![](thumbnails/OpenCountry_FP_image_0244.jpg) | ![](thumbnails/OpenCountry_FN_image_0044.jpg) |
| Coast | ![](thumbnails/Coast_train_image_0146.jpg) | ![](thumbnails/Coast_TP_image_0047.jpg) | ![](thumbnails/Coast_FP_image_0119.jpg) | ![](thumbnails/Coast_FN_image_0244.jpg) |
| Mountain | ![](thumbnails/Mountain_train_image_0185.jpg) | ![](thumbnails/Mountain_TP_image_0279.jpg) | ![](thumbnails/Mountain_FP_image_0124.jpg) | ![](thumbnails/Mountain_FN_image_0047.jpg) |
| Forest | ![](thumbnails/Forest_train_image_0152.jpg) | ![](thumbnails/Forest_TP_image_0081.jpg) | ![](thumbnails/Forest_FP_image_0278.jpg) | ![](thumbnails/Forest_FN_image_0124.jpg) |
