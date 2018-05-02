#JWE
JWE is the chinese word embedding which use CBOW,it add component and radical to the word embedding

#main task
we modify the original Dureader's tensorflow folder which is the main task of Dureader contest

#passage rank
Because the Dureader's baseline use recall to choose the most related paragraph , we use S-net to choose which is the bese paragraph to improve the accuracy

#rc_pr_predict_with_question
combine the main task and passage rank . this floder modify dataset.py in main task and use the pr.json file which is passage rank's result file