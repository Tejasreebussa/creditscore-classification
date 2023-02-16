#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the Data set
df=pd.read_csv("E-Commerce Shipping Data.csv")

#Displaying the head of the dataset, containing 5 rows
df.head()

#Displaying the shape of the dataset
df.shape

#Dispaying the type of the attributes in dataset
df.dtypes

# Non-Graphical EDA - Show Summary to identify unsual values, errors, outliers, etc in data
print(df.describe())

#Data cleaning/processing - Check for Null Values
null_values = df.isnull().sum()
print(null_values)

#Data Cleaning/Processing - Remove Irrelevant Columns
df.drop('ID', axis = 1, inplace = True)

#Data Cleaning/Processing - Remove duplicates
df.drop_duplicates(inplace = True)

#Data Cleaning/Processing - Check for Special Characteristics
regex = re.compile('[^a-zA-Z0-9\s]')
for col in df.columns:
    for value in df[col]:
        if regex.search(str(value)):
            print(f"Special characters found in attribute {col} are: {value}")

#Graphical EDA - Bar plot to show counts of target attribute for Data Balancing Issues
value_counts = df["Reached.on.Time_Y.N"].value_counts()
print(value_counts)
plt.bar(x=value_counts.index, height=value_counts.values)
plt.xlabel("Reached on Time")
plt.ylabel('Count')
plt.title('Value Counts for Target Variable - Reached on Time')
plt.xticks(value_counts.index, ['1', '0'])
plt.show()

#Data Cleaning/Processing - Balancing
class_counts = df['Reached.on.Time_Y.N'].value_counts()
max_count = class_counts.max()
class_0 = df[df['Reached.on.Time_Y.N'] == 0]
class_1 = df[df['Reached.on.Time_Y.N'] == 1]
class_0_oversampled = class_0.sample(n=max_count, replace=True)
df_oversampled = pd.concat([class_0_oversampled, class_1])
df = df_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)

#Bar plot to show counts of target attribute after Data Balancing
value_counts = df["Reached.on.Time_Y.N"].value_counts()
print(value_counts)
plt.bar(x=value_counts.index, height=value_counts.values)
plt.xlabel("Reached on Time")
plt.ylabel('Count')
plt.title('Value Counts for Target Variable - Reached on Time')
plt.xticks(value_counts.index, ['1', '0'])
plt.show()

#Graphical EDA - Show Categorical Columns Data using Pie Chart
categorical_cols = ['Warehouse_block','Mode_of_Shipment','Product_importance','Gender'] 

for col in categorical_cols: 
    plt.figure() 
    df[col].value_counts().plot(kind='pie',autopct='%1.2f%%') 
    plt.title(col) 
    plt.ylabel('Count') 
    plt.legend()
    plt.show()

#Graphical EDA - Show Numerical Columns Data using Hist Plot
df.hist(figsize=(16,16))

#Graphical EDA - Show Gender and Customer Rating Data by grouping with target attribute using Count Plot
columns=['Gender','Customer_rating'] 
plt.figure(figsize = (15, 20))
plot_count = 1
for i in range(len(columns)):
    axis = plt.subplot(5, 2, plot_count)
    sb.countplot(x = columns[i], hue ='Reached.on.Time_Y.N', ax = axis, data = df)
    plt.tight_layout()
    plot_count+=1
plt.show()

#Data Cleaning/Processing - Remove less correlated attributes
df.drop('Gender', axis = 1, inplace = True)
df.drop('Customer_rating', axis = 1, inplace = True)

#Data Cleaning/Processing - Encoding Categorical Values
label_encoder = LabelEncoder()
df['Warehouse_block']=label_encoder.fit_transform(df['Warehouse_block'])
df['Mode_of_Shipment']=label_encoder.fit_transform(df['Mode_of_Shipment'])
df['Product_importance']=label_encoder.fit_transform(df['Product_importance'])

#To Show the summary of dataset after Encoding Categorical data
print(df.describe())

#EDA Graphical - Show the correlation matrix using heatmap
sb.heatmap(df.corr(),annot=True, annot_kws={"size": 5})

# Non-Graphical EDA - Calculate mean, median, mode, std and outliers
outliers = {}
z_scores = np.abs((df - df.mean()) / df.std())

for col in df.columns:
    print()
    print("Column:", col)
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Mode:", df[col].mode())
    print("Standard Deviation:", df[col].std())

    outliers[col] = df[z_scores[col] > 3]
    print("Number of outliers:", len(outliers[col]))

#Graphical EDA - Box plot to show outliers on independent attributes
for col in df.columns[:-1]: 
    fig, ax = plt.subplots()
    ax.boxplot(df[col])
    ax.set_title(col)
    plt.show()

#Graphical EDA - Violin  plot to show outliers in attributes Discount offered and Prior purchases
columns=['Discount_offered','Prior_purchases']
for col in columns:
  sb.violinplot(x=col, y='Reached.on.Time_Y.N', data=df, scale='width', orient='h')
  plt.title('Distribution of %s vs. Reached Time'%col)
  plt.xlabel(col)
  plt.ylabel('Reached on Time (1 = Yes, 0 = No)')
  plt.show()

#Displaying the counts of the attribute before removing Outliers
vc = df['Prior_purchases'].value_counts()
print(vc)

#Data Cleaning/Processing - Remove Outliers
df = df[df['Prior_purchases'] < 6]
df['Prior_purchases'].value_counts()

#Graphical EDA - Scatter plot
sb.scatterplot(data = df, x = 'Weight_in_gms', y= 'Cost_of_the_Product', hue='Reached.on.Time_Y.N')

#Data Cleaning/Processing - Divide into dependent and independent variables
Y=df['Reached.on.Time_Y.N']
X=df.drop('Reached.on.Time_Y.N', axis=1)

print("Independent Variables: ", X.columns)
print("Dependent Variables: ", Y.name)

#Independent Attributes Before Scaling
print(X)

#Data Cleaning/Processing - Feature Scaling
feature_scaling = MinMaxScaler()
feature_scaler = feature_scaling.fit_transform(X)
X = pd.DataFrame(feature_scaler, columns=X.columns)

#Independent Attributes After Scaling
print(X)

#Data Cleaning/Processing - Split the data into test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.01, random_state=0)
print(X_train.shape)
print(X_test.shape)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pickle
def naive_bayes_info():
  nb = GaussianNB()
  nb.fit(X_train, Y_train)
  pickle.dump(nb, open("pickle_files/nb_model", 'wb'))
  Y_pred = nb.predict(X_test)
  print(Y_pred)
  accuracy = metrics.accuracy_score(Y_test, Y_pred)
  print("Accuracy percentage of Gaussian Naive bayes: %f"%(accuracy*100))
  print("Precision Score of Gaussian Naive Bayes: %f"%metrics.precision_score(Y_test, Y_pred))
  print("Recall Score of Gaussian Naive Bayes: %f"% metrics.recall_score(Y_test, Y_pred))
  print("Report:\n ",metrics.classification_report(Y_test, Y_pred))
  print("\nConfusion Matrix:\n")
  conf_matrix = metrics.confusion_matrix(Y_test, Y_pred, labels=nb.classes_)
  disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=nb.classes_)
  disp.plot()
  plt.show()

naive_bayes_info()

def tuned_naive_bayes_info():
  param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
  }
  nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
  nbModel_grid.fit(X_train, Y_train)
  pickle.dump(nbModel_grid, open("pickle_files/tuned_nb_model", 'wb'))
  Y_pred = nbModel_grid.predict(X_test)
  accuracy = metrics.accuracy_score(Y_test, Y_pred)
  print("Accuracy percentage of Gaussian Naive bayes: %f"%(accuracy*100))
  print("Precision Score of Gaussian Naive Bayes: %f"%metrics.precision_score(Y_test, Y_pred))
  print("Recall Score of Gaussian Naive Bayes: %f"% metrics.recall_score(Y_test, Y_pred))
  print("Report:\n ",metrics.classification_report(Y_test, Y_pred))
  print("\nConfusion Matrix:\n")
  conf_matrix = metrics.confusion_matrix(Y_test, Y_pred, labels=nbModel_grid.classes_)
  disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=nbModel_grid.classes_)
  disp.plot()
  plt.show()

tuned_naive_bayes_info()

from sklearn.neighbors import KNeighborsClassifier
def KNN_info(number):
  train_accuracy=[0 for i in range(number+1)]
  test_accuracy=[0 for i in range(number+1)]

  for i in range(1,number+1):
      knn = KNeighborsClassifier(n_neighbors=i)
      knn.fit(X_train, Y_train)
      train_accuracy[i] = knn.score(X_train, Y_train)
      test_accuracy[i] = knn.score(X_test, Y_test)

  plt.plot(range(number+1), test_accuracy, label = 'Testing dataset Accuracy')
  plt.plot(range(number+1), train_accuracy, label = 'Training dataset Accuracy')

  result=0
  index=0
  for i,ele in enumerate(test_accuracy):
    if result<ele:
      result=ele
      index=i

  print("The highest Test accuracy is found at k value %d"%index)
  plt.legend()
  plt.xlabel('k-neighbors')
  plt.ylabel('Accuracy')
  plt.show()
  return index

Optimal_KNN_value=KNN_info(42)

def additional_KNN_info(K_value):
  knn = KNeighborsClassifier(n_neighbors=K_value)
  knn.fit(X_train, Y_train)
  pickle.dump(knn, open("pickle_files/knn_model", 'wb'))
  Y_pred = knn.predict(X_test)
  accuracy = metrics.accuracy_score(Y_test, Y_pred)
  print("Accuracy percentage of KNN: %f"%(accuracy*100))
  print("Precision Score of KNN: %f"%metrics.precision_score(Y_test, Y_pred))
  print("Recall Score of KNN: %f"% metrics.recall_score(Y_test, Y_pred))
  print("Report:\n ",metrics.classification_report(Y_test, Y_pred))
  print("\nConfusion Matrix:\n")
  conf_matrix = metrics.confusion_matrix(Y_test, Y_pred, labels=knn.classes_)
  disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=knn.classes_)
  disp.plot()
  plt.show()

additional_KNN_info(Optimal_KNN_value)

from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def find_best_clusters(df, number):
    
    clusters_inertias = []
    
    for k in range(1, number+1):
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)
        clusters_inertias.append(kmeans_model.inertia_)
    return clusters_inertias

def generate_plot(clusters_inertias, k_values):
  figure = plt.subplots(figsize = (12, 6))
  plt.plot(k_values, clusters_inertias, 'o-', color = 'orange')
  plt.xlabel("Number of Clusters (K)")
  plt.ylabel("Cluster Inertia")
  plt.title("Elbow Plot of KMeans")
  plt.show()

max_value_of_K=12

clusters_inertias = find_best_clusters(df, max_value_of_K)
generate_plot(clusters_inertias, list(range(max_value_of_K)))

def K_means_info(clusters):
  kmeans_model = KMeans(n_clusters = clusters)
  kmeans_model.fit(df)
  pickle.dump(kmeans_model, open("pickle_files/kmeans_model", 'wb'))
  df["clusters"] = kmeans_model.labels_
  return df

no_of_clusters=2
data=K_means_info(no_of_clusters)
seperate_dfs=[]
for i in range(no_of_clusters):
  seperate_dfs.append(df[df["clusters"]==i])

#SVM Algorithm
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

def get_SVM_details(X_train, X_test, Y_train, Y_test,x):
  classify = svm.SVC(kernel='linear')
  classify.fit(X_train, Y_train)
  pickle.dump(classify, open("pickle_files/svm_model", 'wb'))
  Y_pred = classify.predict(X_test)
  if x!=-1:
    print("\nDetails for the cluster no %d:"%x)
  print("Accuracy percentage of SVM: %f"%(metrics.accuracy_score(Y_test, Y_pred)*100))
  print("Precision Score of SVM: %f"%metrics.precision_score(Y_test, Y_pred))
  print("Recall Score of SVM: %f"% metrics.recall_score(Y_test, Y_pred))
  print("Report:\n ",metrics.classification_report(Y_test, Y_pred))
  print("\nConfusion Matrix:\n")
  conf_matrix = metrics.confusion_matrix(Y_test, Y_pred, labels=classify.classes_)
  disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classify.classes_)
  disp.plot()
  plt.show()

def pass_after_split(data,x):
  Y=data['Reached.on.Time_Y.N']
  X=data.drop('Reached.on.Time_Y.N', axis=1)
  X=data.drop('clusters', axis=1)
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
  get_SVM_details(X_train, X_test, Y_train, Y_test,x)

#SVM Details
get_SVM_details(X_train, X_test, Y_train, Y_test, -1)
for i in range(no_of_clusters):
  pass_after_split(seperate_dfs[i],i+1)

#Decison Tree Algorithm
from sklearn.tree import DecisionTreeClassifier

def train_DecisonTree(method):
    clf_gini = DecisionTreeClassifier(criterion = method)
    clf_gini.fit(X_train, Y_train)
    pickle.dump(clf_gini, open("pickle_files/E_DT_model", 'wb'))
    return clf_gini

def get_DecisionTree_details(object):
  Y_pred = object.predict(X_test)
  print("Accuracy percentage of Decision Tree: %f"%(metrics.accuracy_score(Y_test, Y_pred)*100))
  print("Precision Score of Decision Tree: %f"%metrics.precision_score(Y_test, Y_pred))
  print("Recall Score of Decision Tree: %f"% metrics.recall_score(Y_test, Y_pred))
  print("Report:\n ",metrics.classification_report(Y_test, Y_pred))
  print("\nConfusion Matrix:\n")
  conf_matrix = metrics.confusion_matrix(Y_test, Y_pred, labels=object.classes_)
  disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=object.classes_)
  disp.plot()
  plt.show()

#Decison Tree Results
train_gini = train_DecisonTree("gini")
train_entropy = train_DecisonTree("entropy")

print("Trained Using GINI: \n")
#get_DecisionTree_details(train_gini)
print("\nTrained Using Entropy: \n")
get_DecisionTree_details(train_entropy)

