# Web Intelligence

# Author
**Matthias Bartolo 0436103L**

## Preview:
<p align='center'>
  <img src="https://github.com/mbar0075/Web-Intelligence/assets/103250564/de63af6d-9990-4b02-97b4-6c9956b7b74b" style="display: block; margin: 0 auto; width: 40%; height: auto;"></br>
  <img src="https://github.com/mbar0075/Web-Intelligence/assets/103250564/dfecbf07-b0e0-475a-991e-cb297965525d" style="display: block; margin: 0 auto; width: 35%; height: auto;">
  <img src="https://github.com/mbar0075/Web-Intelligence/assets/103250564/f1c8dcde-3677-4dc4-be65-6af91515e8f0"  style="display: block; margin: 0 auto; width: 40%; height: auto;">
</p>

## Description of Task:
This project focused on working with and exploring the fundamental concepts and technologies underlying the Web. It provided a comprehensive understanding of the mathematical principles behind graphs, probabilistic modeling, and data analysis. Additionally, practical applications of various techniques, such as text and link analysis, were covered to effectively analyze the content and structure of the ever-evolving distributed space of the Web.

### 1. Graph Analysis:
The objective of this task was to perform an in-depth analysis and visualization of a football passing network utilizing the Python programming language.

To analyze and visualize a football passing network, a Jupyter Notebook was created using Python. The objective was to utilize the **StatsBomb** data to generate the passing network. The notebook parsed the competition files to obtain a list of competitions and selected specific matches for a chosen competition and season. The lineup and pass data were extracted for each match, and the top 11 players for each team were determined based on playing minutes. The passing network was generated and plotted using the networkX library. Lastly, the passing network for each match was stored in Neo4j Desktop using the nxneo4j library.

Additionally, an analysis was conducted on the underlying structure of the constructed passing network. NetworkX was employed to compute several statistics for each team, including the total number of passes, degree distribution (with a plotted distribution), average path length, and global clustering coefficient. The Jupyter Notebook also addressed any challenges encountered during the computation of these statistics and provided explanations of the approaches taken to overcome them.

```python
import nxneo4j

#Graph Configuration
config = {
'node_label': 'Player',
'relationship_type': 'PASSEDTO',
'identifier_property': 'name'
}

#Creating directed graph, and deleting any previous data
Graph = nxneo4j.DiGraph(driver,config=config)
Graph.delete_all()
```

Furthermore, by utilizing **Neo4j**, specific questions about each team were addressed through direct **Cypher** queries. These queries aimed to identify the most active player in terms of passes, list the top three players with an intermediary role, assess player centrality, and determine the player who received the highest number of passes.

The Jupyter Notebook included the Cypher queries as markdown, providing a clear and structured representation of the queries. Detailed explanations of the parameters used in each query were provided to ensure a comprehensive understanding of the query logic. Any challenges encountered during the querying process were addressed and resolved, and these solutions were described in the notebook to provide a thorough explanation of the process.

This project involved analyzing football passing networks using data analysis, network visualization, and graph database querying. The StatsBomb data was utilized to generate and visualize passing networks in Python. NetworkX was employed to compute statistics and analyze the underlying structure of the networks. Additionally, Neo4j was used as a graph database to store the passing networks and answer specific queries. The project also included tutorials on Neo4j and explored the application of graph analysis in the context of the **Game of Thrones** graph. Overall, it offered a comprehensive exploration of football passing networks and provided insights into graph-based analysis techniques.

<p align='center'>
  <img src="https://github.com/mbar0075/Web-Intelligence/assets/103250564/1775beaa-9aab-40b4-82d7-c0c9fe3ee71f" style="display: block; margin: 0 auto; width: 70%; height: auto;"></br>
</p>


### 2. Text Analysis
This task encompassed a comprehensive text analysis conducted on the **News Category Dataset** available on Kaggle. The dataset consisted of 47,000 news headlines sourced from HuffPost, spanning the period from January 1, 2017, to September 23, 2022. The primary objective was to perform various text analysis tasks using a Jupyter Notebook, yielding valuable insights into the dataset.

In the Jupyter Notebook, the news headline text was meticulously processed. JSON files were parsed, and the relevant text was extracted from each record, focusing on the "headline" and "short description" fields. To facilitate further analysis, lexical analysis techniques were applied to extract individual words, which were subsequently converted to lowercase. Moreover, a standard English stop-word list was employed to eliminate common stop words. Porter's stemmer algorithm was utilized to reduce terms to their respective stems, taking into account ready-made implementations while providing appropriate references.

The next phase involved the calculation of term weights utilizing the TF-IDF approach. Each headline record was treated as an independent document, enabling the computation of term weights specific to each headline.

```python
#Function that utilises all previously built methods to build the term by document matrix
def CalculateVectorSpaceModel(documentList):
    sortedWordFreqList=GetUniqueWords(documentList)
    TFValues=GetTermFrequency(sortedWordFreqList,documentList)
    IDFValues=GetInverseDocumentFrequency(sortedWordFreqList,documentList)
    documentMatrix=GetTFIDF(TFValues,IDFValues)
    return documentMatrix
```

Following this, the highest-weighted n% of terms for each headline category were extracted. This process commenced by calculating the average term weight for all terms within each category. Subsequently, the top n% of terms were identified based on their weights. This compilation of terms, along with their corresponding weights, formed the basis of constructing category keyword clouds. The resulting keyword clouds shed light on the prevalent concepts associated with each news category, striking a balance between inclusivity and conciseness.

To uncover underlying patterns within the dataset, the news headlines were subjected to clustering using the k-means algorithm. A single level of clustering was implemented without incorporating hierarchical structures. The optimal number of clusters, k, was determined based on the specific analysis requirements.

Following the clustering phase, the highest-weighted n% of terms were extracted for each generated cluster. This process entailed computing the average term weight for all terms within each cluster and subsequently selecting the top n% of terms based on their weights. These terms, together with their respective weights, were employed in constructing cluster keyword clouds, offering insights into the prevailing concepts associated with each cluster.

Additionally, detailed information pertaining to each category and cluster was extracted in JSON format. This encompassed the category name, the list of articles within each category, and the highest-weighted terms for each category or cluster. This extracted data served as a valuable resource for the subsequent visualization application.

Throughout the analysis, diligent efforts were made to address any challenges encountered along the way. Suitable solutions were implemented to overcome these obstacles, ensuring the successful completion of the text analysis task.

In order to visualize the obtained results, a simple web application was set up using Flask. This involved setting up Flask as the foundation of the web application and implementing necessary configurations and dependencies to ensure its smooth operation. The JSON files generated during the Text Analysis phase were imported into the application, serving as the source of data for visualization.

```python
import os
from flask import Flask, render_template, json, current_app as app

#Setting up a flask application
app = Flask(__name__)

#Opening the corresponding files and parsing their data as json 
with open('OutputFiles/topWords.json', 'r') as file:
     tpf = json.load(file)

with open('OutputFiles/TFIDF_Category.json', 'r') as file1:
    opf = json.load(file1)   

with open('OutputFiles/TFIDF_Cluster.json', 'r') as file2:
    opf2 = json.load(file2)  
    
#Specifiying the url extensions for each website section
#Default route loads to the main page 
@app.route('/', methods = ['GET', 'POST'])
def loadData():
    return render_template('index.html',tpf=tpf, opf=opf, opf2=opf2)
```

The web application provided various features for visualizing the results:

Firstly, when a document was clicked, the web application displayed its relevant details, including the headline, date, and other pertinent information. Additionally, the corresponding keyword cloud associated with the selected document was presented, enabling users to gain insights into the prominent terms and concepts within that specific document.

Secondly, an interactive bubble chart was implemented to display the list of categories. By clicking on a specific category bubble, users could access the list of documents associated with that category. Furthermore, the keyword cloud corresponding to the selected category was showcased, providing users with a comprehensive overview of the prevalent concepts within that category.

Similarly, a separate interactive bubble chart was implemented to represent the term clusters. By clicking on a cluster bubble, users could explore the list of documents associated with that cluster. Additionally, the keyword cloud specific to the chosen cluster was displayed, offering users a glimpse into the dominant concepts encompassed by that cluster.

The inclusion of interactive visualizations within the web application greatly enhanced the user experience, allowing for effortless navigation and exploration of the text analysis results. These features facilitated a more intuitive and engaging way of interacting with the data. The successful implementation of the web application's functionalities significantly contributed to its overall effectiveness and usefulness. Users were able to access detailed information about individual documents, including headlines, dates, and corresponding keyword clouds, simply by clicking on them. Similarly, the interactive bubble chart showcased categories and clusters, enabling users to explore related documents and their respective keyword clouds with ease.

In addition, the project featured tutorials that provided valuable insights into information retrieval techniques using **TF-IDF (Term Frequency-Inverse Document Frequency)**. These tutorials played a crucial role in enhancing understanding and proficiency in retrieving relevant information from online sources by leveraging TF-IDF.

<p align='center'>
  <img src="https://github.com/mbar0075/Web-Intelligence/assets/103250564/c95f4eac-7091-4559-86ec-41461cdc28cc" style="display: block; margin: 0 auto; width: 70%; height: auto;"></br>
</p>

Overall, the project successfully integrated interactive visualizations, informative tutorials, and advanced information retrieval techniques, resulting in a comprehensive and user-friendly platform for analyzing and exploring text analysis results.


## Deliverables:
The repository includes:<br />
1. **Group Projects** - The directory which holds the Group Project<br />
2. **Individual Projects** - The directory which holds the Individual Projects<br />
 
