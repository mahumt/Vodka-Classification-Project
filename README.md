# Vodka-Classification-Project
Classification methods applied to "Liquour Sales in Iowa State" kaggle dataset (dependant variable: category)

Primary dataset used is from Kaggle: https://www.kaggle.com/residentmario/iowa-liquor-sales, this was compleemnted by dataset about demographic information of the state Iowa which is taken from the Iowa Datacenter: https://www.iowadatacenter.org/browse/ZCTAs.html

An academic project done in a 3-man team. The original project focused on 3 parts: Regression techniques, Classification methods and Unsupervied learning methods: Clustering.

This repository is related to my contribution in the said project: Classification methods on the variable "Category" of vodka. 

The repository does include data preprocessing and exploration codes as well (for clarity sake, and due to usage of two different datasets). But that was done my groupmate: Maria Camila Jimenez Amaya, who deserves the reocgnition.

the dataset from kaggle can be downloaded using Command Line interface (CLI): 

```kaggle datasets download -d residentmario/iowa-liquor-sales```
    
Or by using code in python itself:

```python
import requests
data_url = 'https://www.kaggle.com/residentmario/iowa-liquor-sales' # The direct link to the Kaggle data set
local_filename = "data/vodka/iowa-liquor-sales" # The local path where the data set is saved.
kaggle_info = {'UserName': "yourusername", 'Password': "yourpassword"} # Kaggle Username and Password
r = requests.get(data_url) # Attempts to download the CSV file. Gets rejected because we are not logged in.
r = requests.post(r.url, data = kaggle_info)# ,prefetch = False) # Login to Kaggle and retrieve the data.
f = open(local_filename, 'w') # Writes the data to a local file one chunk at a time
for chunk in r.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory
    if chunk: # filter out keep-alive new chunks
        f.write(chunk)
f.close()
```
