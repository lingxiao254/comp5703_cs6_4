COMP5703 CS 6_4 WHOLE SLIDE IMAGE BREAST CANCER DETECTION BY MACHINE LEARNING

This project is aiming to validate the performance of different commonly used machine learning methods and deep learning methods on detecting IDC from whole slide images and select a better performing algorithm to improve IDC prediction accuracy.
We implemented two machine learning methods Random Forest and Support Vector Machine (SVM), which are commonly used machine learning method and has been proved to be successful in image classification. In addition, we also implemented two deep learning methods, a 3-layer Convolutional Neural Network (CNN) and ResNet-50. We also designed our own deep learning architecture that merge CNN, ResNet-50 and DenseNet together, which is called the merging model.
The results of our project show that the two deep learning models CNN and ResNet50 have better performance than machine learning models (RF and SVM). Though the merging model shows better performance than the machine learning methods, it does not give the best result as expected.


	Data Source: Kaggle Breast Histopathology Images https://www.kaggle.com/paultimothymooney/breast-histopathology-images

	Data Label Preparation: Initial_Data_Extraction_final.ipynb

	Data analysis: EDA_new.ipynb

	Modelling, visualization and model evaluation: 
		RF_final.ipynb
		SVM_final.ipynb
		cnn_final.ipynb
		ResNet50_final.ipynb
		merging_model_ipynb.ipynb
		Visualisation_final.ipynb

	Model comparison: McNemar_ResNet_CNN.ipynb
