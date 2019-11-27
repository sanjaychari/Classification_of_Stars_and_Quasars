import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from datetime import datetime
import tkinter as tk
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

class Node:
	def __init__(self, predicted_class):
		self.predicted_class = predicted_class
		self.feature_index = 0
		self.threshold = 0
		self.left = None
		self.right = None


class Decision_Tree:
	def __init__(self, max_depth=None):
		self.max_depth = max_depth

	def _best_split(self, X, y):
		m = y.size
		if m <= 1:
			return None, None
		num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
		best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
		best_idx, best_thr = None, None
		for idx in range(self.n_features_):
			thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
			num_left = [0] * self.n_classes_
			num_right = num_parent.copy()
			for i in range(1, m):
				c = classes[i - 1]
				num_left[c] += 1
				num_right[c] -= 1
				#gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
				#gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
				sum_left = 0
				for x in range(self.n_classes_):
					sum_left += (num_left[x]/i)**2
				gini_left = 1.0 - sum_left
				sum_right = 0
				for x in range(self.n_classes_):
					sum_right += (num_right[x]/(m-i))**2
				gini_right = 1.0 - sum_right
				gini = (i * gini_left + (m - i) * gini_right) / m
				if thresholds[i] == thresholds[i - 1]:
					continue
				if gini < best_gini:
					best_gini = gini
					best_idx = idx
					best_thr = (thresholds[i] + thresholds[i - 1]) / 2
		return best_idx, best_thr

	def _grow_tree(self, X, y, depth=0):
		num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
		predicted_class = np.argmax(num_samples_per_class)
		node = Node(predicted_class=predicted_class)
		if depth < self.max_depth:
			idx, thr = self._best_split(X, y)
			if idx is not None:
				indices_left = X[:, idx] < thr
				X_left, y_left = X[indices_left], y[indices_left]
				X_right, y_right = X[~indices_left], y[~indices_left]
				node.feature_index = idx
				node.threshold = thr
				node.left = self._grow_tree(X_left, y_left, depth + 1)
				node.right = self._grow_tree(X_right, y_right, depth + 1)
		return node

	def _predict(self, inputs):
		node = self.tree_
		while node.left:
			if inputs[node.feature_index] < node.threshold:
				node = node.left
			else:
				node = node.right
		return node.predicted_class

	def predict(self, X):
		return [self._predict(inputs) for inputs in X]

	def fit(self, X, y):
		self.n_classes_ = len(set(y))
		self.n_features_ = X.shape[1]
		self.tree_ = self._grow_tree(X, y)

def results(df,catalog,cross_validate=False):
	if(cross_validate):
		start=datetime.now()
		X = df.drop(['serial_no', 'class','pred'], axis=1)
		X = X.drop(X.columns[0],axis=1)
		y = df['class']
		scaler = MinMaxScaler(feature_range=(0, 1))
		X = scaler.fit_transform(X)
		#print(X[X.columns[0]])
		clf = Decision_Tree(max_depth=8)
		cv = KFold(n_splits=10, random_state=42, shuffle=False)
		acc_arr=[]
		prec_arr=[]
		rec_arr=[]
		f1_arr=[]
		for train_index, test_index in cv.split(X):
			print("Train Index: ", train_index, "\n")
			print("Test Index: ", test_index)
			X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
			clf.fit(X_train, y_train)
			y_pred=clf.predict(X_test)
			acc_arr.append(accuracy_score(y_test,y_pred))
			prec_arr.append(precision_score(y_test,y_pred))
			rec_arr.append(recall_score(y_test,y_pred))
			f1_arr.append(f1_score(y_test,y_pred))
		end=datetime.now()-start
		accuracy=sum(acc_arr)/len(acc_arr)
		precision=sum(prec_arr)/len(prec_arr)
		recall=sum(rec_arr)/len(rec_arr)
		f1=sum(f1_arr)/len(f1_arr)
		master = tk.Tk()
		master.title("Catalog "+catalog+" Results With 10 Fold Cross Validation and Red Shift")
	else:
		start=datetime.now()
		X = df.drop(['serial_no','spectrometric_redshift', 'class','pred'], axis=1)
		X = X.drop(X.columns[0],axis=1)
		y = df['class']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,stratify=y)
		X_train=X_train.to_numpy()
		y_train=y_train.to_numpy()
		X_test=X_test.to_numpy()
		y_test=y_test.to_numpy()
		clf = Decision_Tree(max_depth=8)
		clf.fit(X_train, y_train)
		end=datetime.now()-start
		y_pred=clf.predict(X_test)
		accuracy=accuracy_score(y_test,y_pred)
		precision=precision_score(y_test,y_pred)
		recall=recall_score(y_test,y_pred)
		f1=f1_score(y_test,y_pred)
		master = tk.Tk()
		master.title("Catalog "+catalog+" Results")
	master.minsize(width=400, height=50)
	w = tk.Label(master, text="Training Time : "+str(end)+"\n\nAccuracy : "+str(accuracy)+"\n\nPrecision : "+str(precision)+"\n\nRecall : "+str(recall)+"\n\nF1 Score : "+str(f1)) 
	w.pack()          
	master.mainloop()

def plot_graphs():
	scores=[]
	times=[]
	depths=list(range(1,15))
	for i in depths:
		print("Depth "+str(i))
		clf = Decision_Tree(max_depth=i)
		sumscore=0
		sumtime=0
		for j in range(1,5):
			print("Catalog "+str(j))
			df = pd.read_csv('MiniProject1_SectionE_G/catalog'+str(j)+'/cat'+str(j)+'.csv')
			start=datetime.now()
			X = df.drop(['serial_no','spectrometric_redshift', 'class','pred'], axis=1)
			X = X.drop(X.columns[0],axis=1)
			y = df['class']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,stratify=y)
			X_train=X_train.to_numpy()
			y_train=y_train.to_numpy()
			X_test=X_test.to_numpy()
			y_test=y_test.to_numpy()
			clf.fit(X_train, y_train)
			end=datetime.now()-start
			y_pred=clf.predict(X_test)
			accuracy=accuracy_score(y_test,y_pred)
			precision=precision_score(y_test,y_pred)
			recall=recall_score(y_test,y_pred)
			f1=f1_score(y_test,y_pred)
			sumscore+=(accuracy+precision+recall+f1)/4
			sumtime+=end.total_seconds()
		scores.append(sumscore/4)
		times.append(sumtime/4)
	x=depths
	plt.title("Cumulative Scores")
	plt.xlabel("Maximum Depth")
	plt.ylabel("Score")
	plt.bar(x,scores)
	plt.show()
	plt.title("Training Times")
	plt.xlabel("Maximum Depth")
	plt.ylabel("Time(in seconds)")
	plt.bar(x,times)
	plt.show()

def cat1():
	df = pd.read_csv('MiniProject1_SectionE_G/catalog1/cat1.csv')
	results(df,'1')

def cat2():
	df = pd.read_csv('MiniProject1_SectionE_G/catalog2/cat2.csv')
	results(df,'2')

def cat3():
	df = pd.read_csv('MiniProject1_SectionE_G/catalog3/cat3.csv')
	results(df,'3')

def cat4():
	df = pd.read_csv('MiniProject1_SectionE_G/catalog4/cat4.csv')
	results(df,'4')

r = tk.Tk() 
r.configure(background='black')
r.attributes("-fullscreen", True) 
r.title('Machine Learning Project') 
w = tk.Label(r, text="-"*1000)
w1 = tk.Label(r, text="MACHINE LEARNING PROJECT")
w1_2 = tk.Label(r, text="CLASSIFICATION OF DATA AS STARS OR QUASARS USING DECISION TREE")
w2 = tk.Label(r, text="-"*1000)
w3 = tk.Label(r, text="Sanjay Chari, PES1201700278")
w4 = tk.Label(r, text="Aditya Shankaran, PES1201700710")
w5 = tk.Label(r, text="Athul Sandosh, PES1201701110")
w6 = tk.Label(r, text="-"*1000)
button1 = tk.Button(r, text='Catalog 1', width=25, command=cat1) 
button2 = tk.Button(r, text='Catalog 2', width=25, command=cat2)
button3 = tk.Button(r, text='Catalog 3', width=25, command=cat3)
button4 = tk.Button(r, text='Catalog 4', width=25, command=cat4)
button5 = tk.Button(r, text='QUIT', width=25, command=r.destroy)
w.place(x=1,y=1)
w1.place(x=590,y=51)
w1_2.place(x=460,y=101)
w2.place(x=1,y=151)
w3.place(x=50,y=251)
w4.place(x=600,y=251)
w5.place(x=1150,y=251)
button1.place(x=50,y=350)
button2.place(x=420,y=350)
button3.place(x=790,y=350)
button4.place(x=1160,y=350)
w6.place(x=1,y=450)
button5.place(x=600,y=600)
r.mainloop() 