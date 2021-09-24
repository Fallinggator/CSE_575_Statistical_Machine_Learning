
# coding: utf-8

# In[1]:


import numpy
import scipy.io
import math
import geneNewData

def main():
    myID='0343'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    
    mean_train0 = []
    std_train0 = []
    mean_train1 = []
    std_train1 = []
    
    for i in train0:
            mean_train0.append(numpy.mean(i))
            std_train0.append(numpy.std(i))
    
    for i in train1:
            mean_train1.append(numpy.mean(i))
            std_train1.append(numpy.std(i))
            
    mean_mean_train0 = numpy.mean(mean_train0)    
    variance_mean_train0 = numpy.var(mean_train0)
    
    mean_std_train0 = numpy.mean(std_train0)    
    variance_std_train0 = numpy.var(std_train0)
    
    mean_mean_train1 = numpy.mean(mean_train1)    
    variance_mean_train1 = numpy.var(mean_train1)
    
    mean_std_train1 = numpy.mean(std_train1)    
    variance_std_train1 = numpy.var(std_train1)
    
#     print(mean_train0)
#     print(std_train0)
#     print(mean_train1)
#     print(std_train1)
    
    print(mean_mean_train0)
    print(variance_mean_train0)
    print(mean_std_train0)
    print(variance_std_train0)
    
    print(mean_mean_train1)
    print(variance_mean_train1)
    print(mean_std_train1)
    print(variance_std_train1)
    
    mean_test0 = []
    std_test0 = []
    mean_test1 = []
    std_test1 = []
    
    for i in test0:
            mean_test0.append(numpy.mean(i))
            std_test0.append(numpy.std(i))
    
    for i in test1:
            mean_test1.append(numpy.mean(i))
            std_test1.append(numpy.std(i))
            
    mean_mean_test0 = numpy.mean(mean_test0)    
    variance_mean_test0 = numpy.var(mean_test0)
    
    mean_std_test0 = numpy.mean(std_test0)    
    variance_std_test0 = numpy.var(std_test0)
    
    mean_mean_test1 = numpy.mean(mean_test1)    
    variance_mean_test1 = numpy.var(mean_test1)
    
    mean_std_test1 = numpy.mean(std_test1)    
    variance_std_test1 = numpy.var(std_test1)

#     print(mean_test0)
#     print(std_test0)
#     print(mean_test1)
#     print(std_test1)
    
#     print(mean_mean_test0)
#     print(variance_mean_test0)
#     print(mean_std_test0)
#     print(variance_std_test0)
    
#     print(mean_mean_test1)
#     print(variance_mean_test1)
#     print(mean_std_test1)
#     print(variance_std_test1)
    
    prob_0 = .5
    prob_1 = .5
    
    # Naive Bayes
    # Function to form PDF
    def p_x_given_y(x, mean, variance):
        p_x_give_y = (1 / (numpy.sqrt(2 * numpy.pi * variance))) * numpy.exp(-(x - mean) ** 2 / (2 * variance))
        return p_x_give_y
    
    # # calculate post prob for image with 0
    post_prob_0_0 = prob_0 * p_x_given_y(mean_test0, mean_mean_train0, variance_mean_train0) * p_x_given_y(std_test0, mean_std_train0, variance_std_train0)
    
    # # calculate post prob for image with 1
    post_prob_1_0 = prob_1 * p_x_given_y(mean_test0, mean_mean_train1, variance_mean_train1) * p_x_given_y(std_test0, mean_std_train1, variance_std_train1)

    # # calculate post prob for image with 0
    post_prob_0_1 = prob_0 * p_x_given_y(mean_test1, mean_mean_train0, variance_mean_train0) * p_x_given_y(std_test1, mean_std_train0, variance_std_train0)
    
    # # calculate post prob for image with 1
    post_prob_1_1 = prob_1 * p_x_given_y(mean_test1, mean_mean_train1, variance_mean_train1) * p_x_given_y(std_test1, mean_std_train1, variance_std_train1)
    
#     print(post_prob_0)
#     print(post_prob_1)
#     test = post_prob_0[2]-post_prob_1[2]
#     print(test)
    compare_0 = numpy.greater(post_prob_0_0, post_prob_1_0)
    compare_1 = numpy.greater(post_prob_1_1, post_prob_0_1)
    
    # Converting the True Values to 1 & False Values to 0
    compare_numeric_0 = compare_0.astype(numpy.int)
    compare_numeric_1 = compare_1.astype(numpy.int)

    # calculating the accuracy for 0
    accuracy_0 = numpy.count_nonzero(compare_numeric_0)/len(compare_numeric_0)
    print(accuracy_0)
    
    # calculating the accuracy for 1
    accuracy_1 = numpy.count_nonzero(compare_numeric_1)/len(compare_numeric_1)
    print(accuracy_1)
#     accuracy_0 = ((np.count_nonzero(np.equal(compare_numeric[0:1028], np.squeeze(testing_set_label)[0:1028]))
#               / np.squeeze(testing_set_label)[0:1028].size)
#               * 100)
#     print('The Accuracy of the Naive Bayes for predicting "0" is ', accuracy_0, "%")

#     # calculating the accuracy for class 1
#     accuracy_1 = ((np.count_nonzero(np.equal(compare_numeric[1028:], np.squeeze(testing_set_label)[1028:]))
#               / np.squeeze(testing_set_label)[1028:].size)
#               * 100)
#     print('The Accuracy of the Naive Bayes for predicting "1" is ', accuracy_1, "%")
    
if __name__ == '__main__':
    main()

