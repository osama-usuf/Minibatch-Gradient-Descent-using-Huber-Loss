from gd import *
import numpy as np

X = np.array(
	   [[ 1.      , 12.96    ,  3.6     ],
	   [ 1.      ,  3.24    ,  1.8     ],
	   [ 1.      , 11.108889,  3.333   ],
	   [ 1.      ,  5.212089,  2.283   ],
	   [ 1.      , 20.548089,  4.533   ],
	   [ 1.      ,  8.311689,  2.883   ],
	   [ 1.      , 22.09    ,  4.7     ],
	   [ 1.      , 12.96    ,  3.6     ],
	   [ 1.      ,  3.8025  ,  1.95    ],
	   [ 1.      , 18.9225  ,  4.35    ],
	   [ 1.      ,  3.359889,  1.833   ],
	   [ 1.      , 15.342889,  3.917   ],
	   [ 1.      , 17.64    ,  4.2     ],
	   [ 1.      ,  3.0625  ,  1.75    ],
	   [ 1.      , 22.09    ,  4.7     ],
	   [ 1.      ,  4.695889,  2.167   ],
	   [ 1.      ,  3.0625  ,  1.75    ],
	   [ 1.      , 23.04    ,  4.8     ],
	   [ 1.      ,  2.56    ,  1.6     ],
	   [ 1.      , 18.0625  ,  4.25    ],
	   [ 1.      ,  3.24    ,  1.8     ],
	   [ 1.      ,  3.0625  ,  1.75    ],
	   [ 1.      , 11.9025  ,  3.45    ],
	   [ 1.      ,  9.406489,  3.067   ],
	   [ 1.      , 20.548089,  4.533   ],
	   [ 1.      , 12.96    ,  3.6     ],
	   [ 1.      ,  3.869089,  1.967   ],
	   [ 1.      , 16.670889,  4.083   ],
	   [ 1.      , 14.8225  ,  3.85    ],
	   [ 1.      , 19.651489,  4.433   ],
	   [ 1.      , 18.49    ,  4.3     ],
	   [ 1.      , 19.954089,  4.467   ],
	   [ 1.      , 11.336689,  3.367   ],
	   [ 1.      , 16.265089,  4.033   ],
	   [ 1.      , 14.691889,  3.833   ],
	   [ 1.      ,  4.068289,  2.017   ],
	   [ 1.      ,  3.485689,  1.867   ],
	   [ 1.      , 23.357889,  4.833   ],
	   [ 1.      ,  3.359889,  1.833   ],
	   [ 1.      , 22.877089,  4.783   ],
	   [ 1.      , 18.9225  ,  4.35    ],
	   [ 1.      ,  3.545689,  1.883   ],
	   [ 1.      , 20.857489,  4.567   ],
	   [ 1.      ,  3.0625  ,  1.75    ],
	   [ 1.      , 20.548089,  4.533   ],
	   [ 1.      , 11.002489,  3.317   ],
	   [ 1.      , 14.691889,  3.833   ],
	   [ 1.      ,  4.41    ,  2.1     ],
	   [ 1.      , 21.464689,  4.633   ],
	   [ 1.      ,  4.      ,  2.      ],
	   [ 1.      , 23.04    ,  4.8     ],
	   [ 1.      , 22.240656,  4.716   ],
	   [ 1.      ,  3.359889,  1.833   ],
	   [ 1.      , 23.357889,  4.833   ],
	   [ 1.      ,  3.003289,  1.733   ],
	   [ 1.      , 23.843689,  4.883   ],
	   [ 1.      , 13.816089,  3.717   ],
	   [ 1.      ,  2.778889,  1.667   ],
	   [ 1.      , 20.857489,  4.567   ],
	   [ 1.      , 18.636489,  4.317   ],
	   [ 1.      ,  4.986289,  2.233   ],
	   [ 1.      , 20.25    ,  4.5     ],
	   [ 1.      ,  3.0625  ,  1.75    ],
	   [ 1.      , 23.04    ,  4.8     ],
	   [ 1.      ,  3.301489,  1.817   ],
	   [ 1.      , 19.36    ,  4.4     ],
	   [ 1.      , 17.363889,  4.167   ],
	   [ 1.      , 22.09    ,  4.7     ],
	   [ 1.      ,  4.272489,  2.067   ],
	   [ 1.      , 22.09    ,  4.7     ],
	   [ 1.      , 16.265089,  4.033   ],
	   [ 1.      ,  3.869089,  1.967   ],
	   [ 1.      , 20.25    ,  4.5     ],
	   [ 1.      , 16.      ,  4.      ],
	   [ 1.      ,  3.932289,  1.983   ],
	   [ 1.      , 25.674489,  5.067   ],
	   [ 1.      ,  4.068289,  2.017   ],
	   [ 1.      , 20.857489,  4.567   ],
	   [ 1.      , 15.077689,  3.883   ],
	   [ 1.      , 12.96    ,  3.6     ],
	   [ 1.      , 17.081689,  4.133   ],
	   [ 1.      , 18.774889,  4.333   ],
	   [ 1.      , 16.81    ,  4.1     ],
	   [ 1.      ,  6.932689,  2.633   ],
	   [ 1.      , 16.540489,  4.067   ],
	   [ 1.      , 24.334489,  4.933   ],
	   [ 1.      , 15.6025  ,  3.95    ],
	   [ 1.      , 20.403289,  4.517   ],
	   [ 1.      ,  4.695889,  2.167   ],
	   [ 1.      , 16.      ,  4.      ],
	   [ 1.      ,  4.84    ,  2.2     ],
	   [ 1.      , 18.774889,  4.333   ],
	   [ 1.      ,  3.485689,  1.867   ],
	   [ 1.      , 23.203489,  4.817   ],
	   [ 1.      ,  3.359889,  1.833   ],
	   [ 1.      , 18.49    ,  4.3     ],
	   [ 1.      , 21.780889,  4.667   ],
	   [ 1.      , 14.0625  ,  3.75    ],
	   [ 1.      ,  3.485689,  1.867   ],
	   [ 1.      , 24.01    ,  4.9     ],
	   [ 1.      ,  6.165289,  2.483   ],
	   [ 1.      , 19.070689,  4.367   ],
	   [ 1.      ,  4.41    ,  2.1     ],
	   [ 1.      , 20.25    ,  4.5     ],
	   [ 1.      , 16.4025  ,  4.05    ],
	   [ 1.      ,  3.485689,  1.867   ],
	   [ 1.      , 22.09    ,  4.7     ],
	   [ 1.      ,  3.179089,  1.783   ],
	   [ 1.      , 23.5225  ,  4.85    ],
	   [ 1.      , 13.564489,  3.683   ],
	   [ 1.      , 22.401289,  4.733   ],
	   [ 1.      ,  5.29    ,  2.3     ],
	   [ 1.      , 24.01    ,  4.9     ],
	   [ 1.      , 19.509889,  4.417   ],
	   [ 1.      ,  2.89    ,  1.7     ],
	   [ 1.      , 21.464689,  4.633   ],
	   [ 1.      ,  5.368489,  2.317   ],
	   [ 1.      , 21.16    ,  4.6     ],
	   [ 1.      ,  3.301489,  1.817   ],
	   [ 1.      , 19.509889,  4.417   ],
	   [ 1.      ,  6.848689,  2.617   ],
	   [ 1.      , 16.540489,  4.067   ],
	   [ 1.      , 18.0625  ,  4.25    ],
	   [ 1.      ,  3.869089,  1.967   ],
	   [ 1.      , 21.16    ,  4.6     ],
	   [ 1.      , 14.190289,  3.767   ],
	   [ 1.      ,  3.674889,  1.917   ],
	   [ 1.      , 20.25    ,  4.5     ],
	   [ 1.      ,  5.139289,  2.267   ],
	   [ 1.      , 21.6225  ,  4.65    ],
	   [ 1.      ,  3.485689,  1.867   ],
	   [ 1.      , 17.363889,  4.167   ],
	   [ 1.      ,  7.84    ,  2.8     ],
	   [ 1.      , 18.774889,  4.333   ],
	   [ 1.      ,  3.359889,  1.833   ],
	   [ 1.      , 19.210689,  4.383   ],
	   [ 1.      ,  3.545689,  1.883   ],
	   [ 1.      , 24.334489,  4.933   ],
	   [ 1.      ,  4.133089,  2.033   ],
	   [ 1.      , 13.935289,  3.733   ],
	   [ 1.      , 17.918289,  4.233   ],
	   [ 1.      ,  4.986289,  2.233   ],
	   [ 1.      , 20.548089,  4.533   ],
	   [ 1.      , 23.203489,  4.817   ],
	   [ 1.      , 18.774889,  4.333   ],
	   [ 1.      ,  3.932289,  1.983   ],
	   [ 1.      , 21.464689,  4.633   ],
	   [ 1.      ,  4.068289,  2.017   ],
	   [ 1.      , 26.01    ,  5.1     ],
	   [ 1.      ,  3.24    ,  1.8     ],
	   [ 1.      , 25.331089,  5.033   ],
	   [ 1.      , 16.      ,  4.      ],
	   [ 1.      ,  5.76    ,  2.4     ],
	   [ 1.      , 21.16    ,  4.6     ],
	   [ 1.      , 12.723489,  3.567   ],
	   [ 1.      , 16.      ,  4.      ],
	   [ 1.      , 20.25    ,  4.5     ],
	   [ 1.      , 16.670889,  4.083   ],
	   [ 1.      ,  3.24    ,  1.8     ],
	   [ 1.      , 15.737089,  3.967   ],
	   [ 1.      ,  4.84    ,  2.2     ],
	   [ 1.      , 17.2225  ,  4.15    ],
	   [ 1.      ,  4.      ,  2.      ],
	   [ 1.      , 14.691889,  3.833   ],
	   [ 1.      , 12.25    ,  3.5     ],
	   [ 1.      , 21.003889,  4.583   ],
	   [ 1.      ,  5.602689,  2.367   ],
	   [ 1.      , 25.      ,  5.      ],
	   [ 1.      ,  3.736489,  1.933   ],
	   [ 1.      , 21.316689,  4.617   ],
	   [ 1.      ,  3.674889,  1.917   ],
	   [ 1.      ,  4.338889,  2.083   ],
	   [ 1.      , 21.003889,  4.583   ],
	   [ 1.      , 11.108889,  3.333   ],
	   [ 1.      , 17.363889,  4.167   ],
	   [ 1.      , 18.774889,  4.333   ],
	   [ 1.      , 20.25    ,  4.5     ],
	   [ 1.      ,  5.841889,  2.417   ],
	   [ 1.      , 16.      ,  4.      ],
	   [ 1.      , 17.363889,  4.167   ],
	   [ 1.      ,  3.545689,  1.883   ],
	   [ 1.      , 21.003889,  4.583   ],
	   [ 1.      , 18.0625  ,  4.25    ],
	   [ 1.      , 14.190289,  3.767   ],
	   [ 1.      ,  4.133089,  2.033   ],
	   [ 1.      , 19.651489,  4.433   ],
	   [ 1.      , 16.670889,  4.083   ],
	   [ 1.      ,  3.359889,  1.833   ],
	   [ 1.      , 19.509889,  4.417   ],
	   [ 1.      ,  4.765489,  2.183   ],
	   [ 1.      , 23.04    ,  4.8     ],
	   [ 1.      ,  3.359889,  1.833   ],
	   [ 1.      , 23.04    ,  4.8     ],
	   [ 1.      , 16.81    ,  4.1     ],
	   [ 1.      , 15.729156,  3.966   ],
	   [ 1.      , 17.918289,  4.233   ],
	   [ 1.      , 12.25    ,  3.5     ],
	   [ 1.      , 19.061956,  4.366   ],
	   [ 1.      ,  5.0625  ,  2.25    ],
	   [ 1.      , 21.780889,  4.667   ],
	   [ 1.      ,  4.41    ,  2.1     ],
	   [ 1.      , 18.9225  ,  4.35    ],
	   [ 1.      , 17.081689,  4.133   ],
	   [ 1.      ,  3.485689,  1.867   ],
	   [ 1.      , 21.16    ,  4.6     ],
	   [ 1.      ,  3.179089,  1.783   ],
	   [ 1.      , 19.070689,  4.367   ],
	   [ 1.      , 14.8225  ,  3.85    ],
	   [ 1.      ,  3.736489,  1.933   ],
	   [ 1.      , 20.25    ,  4.5     ],
	   [ 1.      ,  5.678689,  2.383   ],
	   [ 1.      , 22.09    ,  4.7     ],
	   [ 1.      ,  3.485689,  1.867   ],
	   [ 1.      , 14.691889,  3.833   ],
	   [ 1.      , 11.675889,  3.417   ],
	   [ 1.      , 17.918289,  4.233   ],
	   [ 1.      ,  5.76    ,  2.4     ],
	   [ 1.      , 23.04    ,  4.8     ],
	   [ 1.      ,  4.      ,  2.      ],
	   [ 1.      , 17.2225  ,  4.15    ],
	   [ 1.      ,  3.485689,  1.867   ],
	   [ 1.      , 18.207289,  4.267   ],
	   [ 1.      ,  3.0625  ,  1.75    ],
	   [ 1.      , 20.097289,  4.483   ],
	   [ 1.      , 16.      ,  4.      ],
	   [ 1.      , 16.949689,  4.117   ],
	   [ 1.      , 16.670889,  4.083   ],
	   [ 1.      , 18.207289,  4.267   ],
	   [ 1.      , 15.342889,  3.917   ],
	   [ 1.      , 20.7025  ,  4.55    ],
	   [ 1.      , 16.670889,  4.083   ],
	   [ 1.      ,  5.841889,  2.417   ],
	   [ 1.      , 17.497489,  4.183   ],
	   [ 1.      ,  4.915089,  2.217   ],
	   [ 1.      , 19.8025  ,  4.45    ],
	   [ 1.      ,  3.545689,  1.883   ],
	   [ 1.      ,  3.4225  ,  1.85    ],
	   [ 1.      , 18.344089,  4.283   ],
	   [ 1.      , 15.6025  ,  3.95    ],
	   [ 1.      ,  5.442889,  2.333   ],
	   [ 1.      , 17.2225  ,  4.15    ],
	   [ 1.      ,  5.5225  ,  2.35    ],
	   [ 1.      , 24.334489,  4.933   ],
	   [ 1.      ,  8.41    ,  2.9     ],
	   [ 1.      , 21.003889,  4.583   ],
	   [ 1.      , 14.691889,  3.833   ],
	   [ 1.      ,  4.338889,  2.083   ],
	   [ 1.      , 19.070689,  4.367   ],
	   [ 1.      ,  4.549689,  2.133   ],
	   [ 1.      , 18.9225  ,  4.35    ],
	   [ 1.      ,  4.84    ,  2.2     ],
	   [ 1.      , 19.8025  ,  4.45    ],
	   [ 1.      , 12.723489,  3.567   ],
	   [ 1.      , 20.25    ,  4.5     ],
	   [ 1.      , 17.2225  ,  4.15    ],
	   [ 1.      , 14.569489,  3.817   ],
	   [ 1.      , 15.342889,  3.917   ],
	   [ 1.      , 19.8025  ,  4.45    ],
	   [ 1.      ,  4.      ,  2.      ],
	   [ 1.      , 18.344089,  4.283   ],
	   [ 1.      , 22.724289,  4.767   ],
	   [ 1.      , 20.548089,  4.533   ],
	   [ 1.      ,  3.4225  ,  1.85    ],
	   [ 1.      , 18.0625  ,  4.25    ],
	   [ 1.      ,  3.932289,  1.983   ],
	   [ 1.      ,  5.0625  ,  2.25    ],
	   [ 1.      , 22.5625  ,  4.75    ],
	   [ 1.      , 16.949689,  4.117   ],
	   [ 1.      ,  4.6225  ,  2.15    ],
	   [ 1.      , 19.509889,  4.417   ],
	   [ 1.      ,  3.301489,  1.817   ],
	   [ 1.      , 19.954089,  4.467   ]])

y = np.array([79, 54, 74, 62, 85, 55, 88, 85, 51, 85, 54, 84, 78, 47, 83, 52, 62,
	   84, 52, 79, 51, 47, 78, 69, 74, 83, 55, 76, 78, 79, 73, 77, 66, 80,
	   74, 52, 48, 80, 59, 90, 80, 58, 84, 58, 73, 83, 64, 53, 82, 59, 75,
	   90, 54, 80, 54, 83, 71, 64, 77, 81, 59, 84, 48, 82, 60, 92, 78, 78,
	   65, 73, 82, 56, 79, 71, 62, 76, 60, 78, 76, 83, 75, 82, 70, 65, 73,
	   88, 76, 80, 48, 86, 60, 90, 50, 78, 63, 72, 84, 75, 51, 82, 62, 88,
	   49, 83, 81, 47, 84, 52, 86, 81, 75, 59, 89, 79, 59, 81, 50, 85, 59,
	   87, 53, 69, 77, 56, 88, 81, 45, 82, 55, 90, 45, 83, 56, 89, 46, 82,
	   51, 86, 53, 79, 81, 60, 82, 77, 76, 59, 80, 49, 96, 53, 77, 77, 65,
	   81, 71, 70, 81, 93, 53, 89, 45, 86, 58, 78, 66, 76, 63, 88, 52, 93,
	   49, 57, 77, 68, 81, 81, 73, 50, 85, 74, 55, 77, 83, 83, 51, 78, 84,
	   46, 83, 55, 81, 57, 76, 84, 77, 81, 87, 77, 51, 78, 60, 82, 91, 53,
	   78, 46, 77, 84, 49, 83, 71, 80, 49, 75, 64, 76, 53, 94, 55, 76, 50,
	   82, 54, 75, 78, 79, 78, 78, 70, 79, 70, 54, 86, 50, 90, 54, 54, 77,
	   79, 64, 75, 47, 86, 63, 85, 82, 57, 82, 67, 74, 54, 83, 73, 73, 88,
	   80, 71, 83, 56, 79, 78, 84, 58, 83, 43, 60, 75, 81, 46, 90, 46, 74])

def huber(x):
	myval = np.array(x)
	result = np.where(np.abs(myval) <= 1 , 0.5*myval**2, np.abs(myval)-0.5)
	return result.sum()

def test_q1():
	# w = np.ndarray([0.3, 1.2, 1.7],1)
	w=np.array([0.3,1.2,1.7])
	# print(w.shape)
	# print(w)
	indices = np.array([1, 3, 5])
	lamb = 0.8
	print(question1_loss(w, X, y, indices, lamb, huber))
	# assert question1_loss(w,X,y,indices,lamb,huber) == 136.68026640000002
def test_q2():
	w = np.array([0.125, 0.5, 0.75])
	ind = np.array([1, 2, 5])
	lamb = 0.001
	print(question2_grad(w,X,y,ind,lamb,huber))

def test_q3():
	w = np.array([0.125,0.5,  0.75 ])
	ind = np.array([1, 2, 5])
	lamb = 0.001
	eta = 0.1
	print(question3_update(w,X,y,ind,lamb,eta,huber))

def makeshuffler(seed =3) :
	prng = np.random.RandomState(seed)
	def shuffle (X, y):
		indices = np.arange(y.size)
		prng.shuffle(indices)
		return X[indices], y[indices]
	return shuffle

shuffle = makeshuffler()

def test_q4():
	w = np.array([0.125,0.5,  0.75 ])
	lamb = 0.1
	eta = 1e-7
	n = 10
	mbatch = 32
	print(question4_n_updates(w, X, y, lamb, eta, mbatch, n, huber, shuffle))

def test_q5():
	w = np.array([0.1, 0.2, 0.8])
	mbatch=10
	lamb=0.2
	eta=1e-09
	nepochs=100
	print(question5_nepochs(w, X, y, lamb, eta, mbatch, nepochs, huber, shuffle))

def test_q6():
	'''Expected: [1.00044990950414, 0.6063842859060459, 1.9016211671054108],'''
	w = np.array([1.,0.6,1.9])
	mbatch=15
	lamb=0.2
	eta=1e-10
	nepochs=150
	epsilon=0.01
	print(question6_sgd(w, X, y, lamb, eta, epsilon, mbatch, nepochs, shuffle))

test_q6()