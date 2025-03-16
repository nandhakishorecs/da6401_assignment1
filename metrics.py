# ------------------------------------------ Classification Metrics --------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA6401 Introduction to Deep Learning Assignment 1
#
#   This file contains code calculate various metrics to measure the goodness of a classifier algorithm.
#	The metrics are Precision, Recal, F1_Score and Accuracy using weighted, micro and macro averages.
#
#   Description of functions in the class 'Metrics'
#
#       - accuracy_score (y_true, y_pred)
#           *   returns the accuracy (ratio of the correct predictions to the total predictions)
#
#       - prconfusion_matrix(y_true, y_pred)
#           *   Converts the given prediction vector into confusion matrix by arranging the predictions into
#				True Positives, True Negatives, False Positives, False Negatives. 
#           *   Precision is defined as the ratio of True Positives to the sum of True Positives and False 
#				Positives 
#			* 	Recal is defined as the ratio of True Positive to the ratio of sum of True Positive and False
#				Negatives
#           *	F2 Score is defined as the Harmonic mean of Precision and Recal. 
#
#		- classification_report(y_true, y_pred, labels)
#			* 	Returns a string, which contains the support values for Precision, Recal and F1 Scores for a
#				classifier. 
# -------------------------------------------------------------------------------------------------------------


# ------------------------------------------- Importing Libraries ----------------------------------------
import numpy as np

class Metrics:
	# ----------------------- Helper Function to compute accuracy of a classifier -----------------------
	def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		accuracy = np.mean(y_true == y_pred)
		return accuracy

	# ----------------------- Helper Function to compute confusion matrix from predictions --------------
	def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
		matrix = np.zeros((len(labels), len(labels)), dtype=int)
		label_to_index = {label: index for index, label in enumerate(labels)}
		for true, predection in zip(y_true, y_pred):
				matrix[label_to_index[true]][label_to_index[predection]] += 1
		return matrix

	# ------------------- Calculating Precision, Recal and F1 Score with the Confusion matrix -----------
	def precision_recall_f1_support(self, conf_matrix: np.ndarray) -> dict:
		metrics = {}
		for i in range(len(conf_matrix)):
			TP = conf_matrix[i, i]
			FP = conf_matrix[:, i].sum() - TP
			FN = conf_matrix[i, :].sum() - TP
			support = conf_matrix[i, :].sum()

			# Precision Calculation
			if ((TP + FP) > 0): 
				precision = TP / (TP + FP)
			else: 
				precision = 0 

			# Recal Calculation
			if ((TP + FN) > 0):
				recall = TP / (TP + FN) 
			else:
				recall = 0

			# F1 Score Calculation
			if ((precision + recall) > 0):
				f1_score = 2 * precision * recall / (precision + recall)  
			else:
				f1_score = 0
			
			metrics[i] = {
				'precision': precision,
				'recall': recall,
				'f1_score': f1_score,
				'support': support
			}
		return metrics

	def classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray = None) -> str:
		if labels is None:
			labels = np.unique(np.concatenate((y_true, y_pred)))
		
		conf_matrix = self.confusion_matrix(y_true, y_pred, labels)
		metrics = self.precision_recall_f1_support(conf_matrix)
		
		# Formatting the report
		report = "              precision    recall  f1_score   support\n\n"
		for i, label in enumerate(labels):
			report += f"{label: >12} "
			report += f"{metrics[i]['precision']: >9.2f} "
			report += f"{metrics[i]['recall']: >9.2f} "
			report += f"{metrics[i]['f1_score']: >9.2f} "
			report += f"{metrics[i]['support']: >9}\n"
		
		# Calculate and add averages
		total_support = sum(m['support'] for m in metrics.values())
		avg_precision = sum(m['precision'] * m['support'] for m in metrics.values()) / total_support
		avg_recall = sum(m['recall'] * m['support'] for m in metrics.values()) / total_support
		avg_f1 = sum(m['f1_score'] * m['support'] for m in metrics.values()) / total_support
		
		report += "\n   macro avg"
		report += f"{np.mean([m['precision'] for m in metrics.values()]): >10.2f} "
		report += f"{np.mean([m['recall'] for m in metrics.values()]): >10.2f} "
		report += f"{np.mean([m['f1_score'] for m in metrics.values()]): >10.2f} "
		report += f"{total_support: >9}\n"
		
		report += "weighted avg"
		report += f"{avg_precision: >10.2f} "
		report += f"{avg_recall: >10.2f} "
		report += f"{avg_f1: >10.2f} "
		report += f"{total_support: >9}\n"
		
		return report

# COMPLETED