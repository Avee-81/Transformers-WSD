
# Transformers-WSD

This repository contains the code to a comparative study on WSD using Nine pre-trained Transformer Models implemented in the **HuggingFace** framework. These models are:

 1. BERT
 2. CTRL
 3. DistilBERT
 4. OpenAIGPT
 5. OpenAIGPT2
 6. TransformerXL
 7. XLNet
 8. ALBERT
 9. ELECTRA

## Requirements:

 1. [Pytorch](https://pytorch.org/)
 2. [Pytorch-Transformers](https://github.com/huggingface/transformers)
 3. [Scikit-learn](https://scikit-learn.org/)

To generate the results for each model, first make sure adhere to the structure of the directory required to complete the execution of code without any problem. 

Store the SenseEval 2 and SensEval 3 train and test data in a folder name "data". 

Make 2 folders named: "results" and "pickles".

 - In "pickles/" make nine more sub-folders whose name should be same as the name of the nine models written above(take care of the case of the model names).
 - Do the same thing mentioned above in "results/" and make one more folder name "analysis".  In each of these model folders, make four more sub-folders:
		 - CSV_SE3
		 - CSV_SE2
		 - XML_SE2
		 - XML_SE3 

Once you have ensured that the structure is correct, you can proceed with the execution.

To generate the predictions for each model, run the following command:

    python3 model.py

Once the execution is over, all the results shall get generated. "pickles" folder stores the CWEs obtained from training data of SE2 and SE3 from each of the nine models. These are used to generate the t-SNE plots.

All the XML files generated as predictions by each of the nine models is stored at "results/'model-name'/XML_SE2 or XML_SE3". To evaluate the performance, we make use of the evaluation framework provided by [UFSAC](https://github.com/getalp/UFSAC).

To generate the t-SNE plots, first make sure that all the execution of the model.py file has been completed. Run the following command to generate the t-SNE plots:

    python3 Tsne_plotter.py

t-SNE plots for each of the models will get stored at "results/'model-name'/" location. This script only genearates the t-SNE plots for the word 'bank' obtained from SE3-training data.

Lastly, the file used to generate Data Statistics is "Analysis.py". Run the following command to generate these statistics.

    python3 Analysis.py

Results are stored as a CSV file in "results/analysis/" folder.



