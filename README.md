# 🔬 SemCluster
this is SemCluster project.

## 📦 Packages Needed  
```
python==3.6
tensorflow==1.14.0
keras==2.2.4  
opencv==3.3.1
pandas==0.20.3
numpy==1.16.4
gensim==3.8.0
jieba==0.42.1  
scipy==1.5.2
scikit-learn==0.20.3
baidu-aip==2.2.18.0
apted==1.0.3
```

💡 baidu-aip is for OCR engine, which is replaceable.

⚠️ OCR key: configure.py, users should place their keys.

## ⚙️ Function Illustration

#### 🚀 main.py

Call the main function of the image processing part image_main and the main function of the text processing part text_main, and complete the entire processing flow by entering the path of the image storage folder, the label file path of the image and text, and setting the xml file storage path. Output the final clustering results and indicators.

#### 🎯 cluster.py

Perform semi-supervised and unsupervised clustering according to the obtained st, ct, p, r and total feature matrix, and get the clustering results.

#### 🖼️ image_main.py

The main function file of the image feature processing part. Get and return the feature matrix of st and ct.

#### 🌲 st_feature_extra.py

Through functions such as flat_bounding, process_bounding, and layout, the image is processed, segmented and optimized to obtain the segmented structure information (st feature) of the image. Return this structure information as a nested string for invocation by the st_sim_matrix.py file.

#### 📐 st_sim_matrix.py

According to the obtained image structure information, the getSTdis function uses the APTED tree edit distance algorithm to calculate the tree edit distance between images, normalizes it to the distance matrix of the st feature, and returns the st distance matrix.

#### 🧩 ct_feature_extra_sim_matrix.py

The process function first matches the two images according to the set threshold, extracts the ct features of the images through the vgg16 model (pre-saved model), and obtains a list of distances between matched controls, with their average value as the Final ct distance. Finally, the ct distances between all the reported images are stored in the form of a ct distance matrix and returned by the getCTdis function.

#### 📝 text_main.py

The main function file of the text feature processing part. Get and return the eigenmatrix of p and r.

#### 🔍 report_feature_extra.py

The widget_problem_text_sim function calculates text sim between text extracted by OCR from widget images and problem widget text analyzed by text_feature_extra module.

distinguish_widgets function find problem widget image from all widgets by problem_widget text info.

The extract_report_feature function is the main process of text feature extraction.

#### 📊 report_sim_matrix.py

The feature matrix of the corresponding p and r features is calculated from the obtained text features, and returned.
