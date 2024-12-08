#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the required liberaries
from neo4j import GraphDatabase
import pandas as pd
import time
from py2neo import Graph
import math
import numpy as np

driver=GraphDatabase.driver(uri="bolt://localhost:7687", auth=("neo4j", "ludhiana43"))
session=driver.session()


# In[2]:


#Loading the data from SDT database
df = pd.read_csv('datasets/POI_1.csv')
df.dropna(subset=['Theme', 'Sub Theme', 'Feature Name', 'Co-ordinates'], inplace=True)


# In[3]:


q1 = """LOAD CSV WITH HEADERS FROM 'file:///POI_1.csv' AS f1
    create(N:bulkcsv{Theme:f1.Theme, Name:f1.FeatureName, Coordinates: f1.Coordinates})"""

record=session.run(q1)


# In[4]:


#Creating the Knowledge graph
def create_knowledge_graph(tx, theme, subtheme, feature_name, coordinates):
    # Building nodes
    tx.run("MERGE (t:Theme {name: $theme})", theme=theme)
    
    tx.run("MERGE (st:SubTheme {name: $subtheme})", subtheme=subtheme)
    
    tx.run("MERGE (f:Feature {name: $feature_name, coordinates: $coordinates})", 
           feature_name=feature_name, coordinates=coordinates)
    
    # Creating relationships
    tx.run("""
    MATCH (t:Theme {name: $theme}), (st:SubTheme {name: $subtheme})
    MERGE (t)-[:HAS_SUB_THEME]->(st)
    """, theme=theme, subtheme=subtheme)
    
    tx.run("""
    MATCH (st:SubTheme {name: $subtheme}), (f:Feature {name: $feature_name})
    MERGE (st)-[:HAS_NAME]->(f)
    """, subtheme=subtheme, feature_name=feature_name)


# In[5]:


def main():
    with driver.session() as session:
        for index, row in df.iterrows():
            session.write_transaction(create_knowledge_graph, 
                                      row['Theme'], 
                                      row['Sub Theme'], 
                                      row['Feature Name'], 
                                      row['Co-ordinates'])

    driver.close()
    
if __name__ == "__main__":
    main()


# In[6]:


#getting all the themes present in the database to create them as nodes
def fetch_themes(tx):
    result = tx.run("MATCH (t:Theme) RETURN t.name AS theme")
    return [record["theme"] for record in result]

with driver.session() as session:
    themes = session.read_transaction(fetch_themes)
    print("Themes:", themes)


# In[7]:


def fetch_subthemes(tx):
    result = tx.run("""
    MATCH (t:Theme)-[:HAS_SUB_THEME]->(st:SubTheme)
    RETURN t.name AS theme, st.name AS subtheme
    """)
    return [{"theme": record["theme"], "subtheme": record["subtheme"]} for record in result]

with driver.session() as session:
    subthemes = session.read_transaction(fetch_subthemes)
    for subtheme in subthemes:
        print(f"Theme: {subtheme['theme']}, SubTheme: {subtheme['subtheme']}")


# In[8]:


def fetch_features(tx):
    result = tx.run("""
    MATCH (st:SubTheme)-[:HAS_FEATURE]->(f:Feature)
    RETURN st.name AS subtheme, f.name AS feature, f.coordinates AS coordinates
    """)
    return [{"subtheme": record["subtheme"], "feature": record["feature"], "coordinates": record["coordinates"]} for record in result]

with driver.session() as session:
    features = session.read_transaction(fetch_features)
    


# In[9]:


#Adding values in Knowledge graph
graph = Graph("bolt://localhost:7687", auth=("neo4j", "ludhiana43"))

def populate_knowledge_graph(tx, theme, subtheme, feature_name, coordinates):
    tx.run("MERGE (t:Theme {name: $theme})", theme=theme)
    tx.run("MERGE (st:SubTheme {name: $subtheme})", subtheme=subtheme)
    tx.run("MERGE (f:Feature {name: $feature_name, coordinates: $coordinates})", 
           feature_name=feature_name, coordinates=coordinates)
    tx.run("""
    MATCH (t:Theme {name: $theme}), (st:SubTheme {name: $subtheme})
    MERGE (t)-[:HAS_SUB_THEME]->(st)
    """, theme=theme, subtheme=subtheme)
    tx.run("""
    MATCH (st:SubTheme {name: $subtheme}), (f:Feature {name: $feature_name})
    MERGE (st)-[:HAS_FEATURE]->(f)
    """, subtheme=subtheme, feature_name=feature_name)


df = pd.DataFrame({
    'Theme': ['Theme1', 'Theme2'],
    'Sub Theme': ['SubTheme1', 'SubTheme2'],
    'Feature Name': ['Feature1', 'Feature2'],
    'Co-ordinates': ['Coord1', 'Coord2']
})

# Running the knowledge graph
tx = graph.begin()
for index, row in df.iterrows():
    populate_knowledge_graph(tx, row['Theme'], row['Sub Theme'], row['Feature Name'], row['Co-ordinates'])
tx.commit()

# Cypher query
result = graph.run("MATCH (n)-[r]->(m) RETURN n, r, m").data()



# In[10]:


def fetch_themes(tx):
    result = tx.run("MATCH (t:Theme) RETURN t.name AS theme")
    return [record["theme"] for record in result]

# Fetch SubThemes for a specific theme
def fetch_subthemes_for_theme(tx, theme):
    result = tx.run("""
    MATCH (t:Theme {name: $theme})-[:HAS_SUB_THEME]->(st:SubTheme)
    RETURN t.name AS theme, st.name AS subtheme
    """, theme=theme)
    return [{"theme": record["theme"], "subtheme": record["subtheme"]} for record in result]

# Fetch Features for a specific subtheme
def fetch_features_for_subtheme(tx, subtheme):
    result = tx.run("""
    MATCH (st:SubTheme {name: $subtheme})-[:HAS_FEATURE]->(f:Feature)
    RETURN st.name AS subtheme, f.name AS feature, f.coordinates AS coordinates
    """, subtheme=subtheme)
    return [{"subtheme": record["subtheme"], "feature": record["feature"], "coordinates": record["coordinates"]} for record in result]


# In[11]:


#Search based query based on themes present as nodes in Knowledge graph
def interactive_query():
    with driver.session() as session:
        themes = session.read_transaction(fetch_themes)
        print("Available Themes:")
        for theme in themes:
            print(theme)
        
        chosen_theme = input("Please enter a theme from the list above: ")
        
        subthemes = session.read_transaction(fetch_subthemes_for_theme, chosen_theme)
        print(f"SubThemes for Theme '{chosen_theme}':")
        for subtheme in subthemes:
            print(subtheme['subtheme'])
        
        for subtheme in subthemes:
            features = session.read_transaction(fetch_features_for_subtheme, subtheme['subtheme'])
            print(f"\nFeatures for SubTheme '{subtheme['subtheme']}':")
            for feature in features:
                print(f"Feature: {feature['feature']}, Coordinates: {feature['coordinates']}")

interactive_query()


# In[12]:



def fetch_spatial_features(tx):
    result = tx.run("""
    MATCH (f:Feature)
    RETURN f.name AS feature, f.coordinates AS coordinates
    """)
    return [{"feature": record["feature"], "coordinates": record["coordinates"]} for record in result]

with driver.session() as session:
    features = session.read_transaction(fetch_spatial_features)


# In[13]:


#getting all the attributes
query = """
MATCH (t:Theme)-[:HAS_SUB_THEME]->(st:SubTheme)-[:HAS_FEATURE]->(f:Feature)
RETURN t.name AS theme, st.name AS subtheme, f.name AS feature, f.coordinates AS coordinates
"""
graph_data = graph.run(query).data()


for record in graph_data:
    print(record)


# In[14]:


#function to create Rectangle Range for the Faceted Query
def is_within_bounds(lat, lon, lat_min, lon_min, lat_max, lon_max):
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

#function to get all the features in the range  
def find_features_in_rectangle(graph_data, lat_min, lon_min, lat_max, lon_max):
    start_time = time.time()
    features_in_rectangle = []
    
    for entry in graph_data:
        try:
            lat, lon = map(float, entry['coordinates'].split(', '))
            
            if is_within_bounds(lat, lon, lat_min, lon_min, lat_max, lon_max):
                features_in_rectangle.append(entry['feature'])
        except ValueError:
            print(f"Invalid Coordinates: {entry['coordinates']}")
    end_time = time.time()

    duration = end_time - start_time
    print(f"Query execution time: {duration:.4f} seconds")
    
    return features_in_rectangle

# Test co-ordinates
lat_min = -37.820  
lon_min = 144.940  
lat_max = -37.800  
lon_max = 144.970  

features = find_features_in_rectangle(graph_data, lat_min, lon_min, lat_max, lon_max)

#printing the result
print("Features within the rectangle:")
for feature in features:
    print(feature)


# In[15]:


#Search Query by applying a filter of Range
def is_within_bounds(lat, lon, lat_min, lon_min, lat_max, lon_max):
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

def find_features_in_rectangle_and_theme(graph_data, theme, lat_min, lon_min, lat_max, lon_max):
    start_time = time.time()
    
    features_in_rectangle_and_theme = []
    
    for entry in graph_data:
        try:
            lat, lon = map(float, entry['coordinates'].split(', '))
            
            if is_within_bounds(lat, lon, lat_min, lon_min, lat_max, lon_max) and entry['theme'] == theme:
                features_in_rectangle_and_theme.append(entry['feature'])
        except ValueError:
            print(f"Invalid Coordinates: {entry['coordinates']}")  
    
    end_time = time.time()

    duration = end_time - start_time
    print(f"Query execution time: {duration:.4f} seconds")
    
    return features_in_rectangle_and_theme

# Input
desired_theme = input("Please enter the theme you are interested in: ")

# Test
lat_min = -37.830  
lon_min = 144.940  
lat_max = -37.800  
lon_max = 144.970  


features = find_features_in_rectangle_and_theme(graph_data, desired_theme, lat_min, lon_min, lat_max, lon_max)
print("Features within the rectangle and matching the theme:")
for feature in features:
    print(feature)


# In[16]:


#function to calculate Distance
def calculate_distance(lat1, lon1, lat2, lon2):
    # Haversine formula to calculate the great-circle distance between two points on the Earth
    R = 6371  # Radius of the Earth in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# In[17]:


#KNN query on knowledge graph using user's location as reference point
def knn_query(graph, query_lat, query_lon, k):
    start_time = time.time()
    query = """
    MATCH (f:Feature)
    RETURN f.name AS feature, f.coordinates AS coordinates
    """
    features = graph.run(query).data()
    
    distances = []
    for feature in features:
        try:
            lat, lon = map(float, feature['coordinates'].split(', '))
            distance = calculate_distance(query_lat, query_lon, lat, lon)
            distances.append((feature['feature'], distance))
        except ValueError:
            print(f"Skipping unknown coordinates: {feature['coordinates']}")

    distances.sort(key=lambda x: x[1])
    nearest_neighbors = distances[:k]
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Query execution time: {duration:.4f} seconds")
    
    return nearest_neighbors
#test co-ordinates
query_lat = -37.820 
query_lon = 144.950  
k = 3  

nearest_neighbors = knn_query(graph, query_lat, query_lon, k)

#printing the result
print(f"The {k} nearest neighbors are:")
for feature, distance in nearest_neighbors:
    print(f"Feature: {feature}, Distance: {distance:.2f} km")


# In[18]:


#combination of range and KNN Query
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def knn_query_for_theme(graph, query_lat, query_lon, theme, k=5):
    start_time = time.time()
    
    query = """
    MATCH (t:Theme {name: $theme})-[:HAS_SUB_THEME]->(st:SubTheme)-[:HAS_FEATURE]->(f:Feature)
    RETURN f.name AS feature, f.coordinates AS coordinates
    """
    features = graph.run(query, theme=theme).data()
    
    distances = []
    for feature in features:
        try:
            lat, lon = map(float, feature['coordinates'].split(', '))
            distance = calculate_distance(query_lat, query_lon, lat, lon)
            distances.append((feature['feature'], distance))
        except ValueError:
            print(f"Skipping invalid coordinates: {feature['coordinates']}")
    
    distances.sort(key=lambda x: x[1])
    nearest_neighbors = distances[:k]
    
    end_time = time.time()

    duration = end_time - start_time
    print(f"Query execution time: {duration:.4f} seconds")
    
    return nearest_neighbors

query_lat = -37.820  # Example latitude
query_lon = 144.950  # Example longitude
theme = input("Please enter a theme: ") 
k = 3 

nearest_neighbors = knn_query_for_theme(graph, query_lat, query_lon, theme, k)

print(f"The {k} nearest neighbors in the '{theme}' theme are:")
for feature, distance in nearest_neighbors:
    print(f"Feature: {feature}, Distance: {distance:.2f} km")


# In[19]:


#Introducing Embeddings

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

graph = Graph(uri="bolt://localhost:7687", auth=("neo4j", "ludhiana43"))

def get_theme_embeddings(graph):
    query = "MATCH (t:Theme) RETURN t.name AS theme"
    themes = graph.run(query).data()
    
    theme_embeddings = {}
    
    for theme in themes:
        theme_name = theme['theme']
        # Generate embedding for each theme
        embedding = model.encode(theme_name)
        theme_embeddings[theme_name] = embedding
    
    return theme_embeddings

def find_most_similar_theme(user_input, theme_embeddings):
    # Generate embedding for the user's input theme
    user_input_embedding = model.encode(user_input)
    
    # Calculate cosine similarity between the user input and all existing themes
    similarities = {}
    for theme, embedding in theme_embeddings.items():
        similarity = cosine_similarity([user_input_embedding], [embedding])[0][0]
        similarities[theme] = similarity
    
    # Find the most similar theme
    most_similar_theme = max(similarities, key=similarities.get)
    similarity_score = similarities[most_similar_theme]
    
    return most_similar_theme, similarity_score

# Generate embeddings for all existing themes
theme_embeddings = get_theme_embeddings(graph)

# Example usage
user_input_theme = input("Please enter a new theme: ")
most_similar_theme, similarity_score = find_most_similar_theme(user_input_theme, theme_embeddings)

print(f"The most similar theme to '{user_input_theme}' is '{most_similar_theme}' with a similarity score of {similarity_score:.2f}.")


# In[20]:


#KNN Query using Similarity embeddings
def knn_query_for_theme(graph, query_lat, query_lon, theme, k=5):
    
    query = """
    MATCH (t:Theme {name: $theme})-[:HAS_SUB_THEME]->(st:SubTheme)-[:HAS_FEATURE]->(f:Feature)
    RETURN f.name AS feature, f.coordinates AS coordinates
    """
    features = graph.run(query, theme=theme).data()
    
    distances = []
    for feature in features:
        try:
            lat, lon = map(float, feature['coordinates'].split(', '))
            distance = calculate_distance(query_lat, query_lon, lat, lon)
            distances.append((feature['feature'], distance))
        except ValueError:
            print(f"Skipping invalid coordinates: {feature['coordinates']}")
    
    distances.sort(key=lambda x: x[1])
    
    nearest_neighbors = distances[:k]
    
    
    return nearest_neighbors


theme_embeddings = get_theme_embeddings(graph)
user_input_theme = input("Please enter a new theme: ")
start_time = time.time()
most_similar_theme, similarity_score = find_most_similar_theme(user_input_theme, theme_embeddings)

print(f"The most similar theme to '{user_input_theme}' is '{most_similar_theme}' with a similarity score of {similarity_score:.2f}.")

query_lat = -37.820 
query_lon = 144.950  
k = 3  

nearest_neighbors = knn_query_for_theme(graph, query_lat, query_lon, most_similar_theme, k)

end_time = time.time()
duration = end_time - start_time
print(f"Query execution time: {duration:.4f} seconds")
    
print(f"The {k} nearest neighbors in the '{most_similar_theme}' theme are:")
for feature, distance in nearest_neighbors:
    print(f"Feature: {feature}, Distance: {distance:.2f} km")


# In[21]:


#Comparison of search query in Cypher and SQL model
import time
import matplotlib.pyplot as plt
from py2neo import Graph
import psycopg2

def execute_cypher_query():

    start_time = time.time()
    graph = Graph(uri="bolt://localhost:7687", auth=("neo4j", "ludhiana43"))

    # Define the Cypher query
    cypher_query = """
    MATCH (f:Feature)
    RETURN f.name AS feature, f.coordinates AS coordinates
    """
    
    

    results = graph.run(cypher_query).data()

    end_time = time.time()
    
    # Calculate the execution time
    execution_time = end_time - start_time
    print(f"Cypher Query Execution Time (Neo4j): {execution_time:.4f} seconds")
    
    return execution_time, results

def execute_sql_query():
    start_time = time.time()
    conn = psycopg2.connect(database="mydb", user="postgres", password="postgres", host="localhost", port="5432")

    sql_query = """
    SELECT p1.feature, p1.coordinates
    FROM POI p1
    JOIN POI p2 ON p1.feature = p2.feature
    UNION ALL
    SELECT p1.feature, p1.coordinates
    FROM POI_2 p1
    JOIN POI_2 p2 ON p1.feature = p2.feature;
    """
    
    
    
    cursor = conn.cursor()
    cursor.execute(sql_query)
    results = cursor.fetchall()
    
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"SQL Query Execution Time (PostgreSQL): {execution_time:.4f} seconds")
    
    cursor.close()
    conn.close()
    
    return execution_time, results

def compare_queries():
    cypher_time, cypher_results = execute_cypher_query()

    sql_time, sql_results = execute_sql_query()

    queries = ['Cypher (Neo4j)', 'SQL (PostgreSQL)']
    times = [cypher_time, sql_time]

    plt.figure(figsize=(10, 6))
    plt.bar(queries, times, color=['blue', 'orange'])
    plt.ylabel('Execution Time (seconds)')
    plt.title('Query Execution Time: Neo4j vs PostgreSQL')
    plt.show()

    print("\nCypher Results (Neo4j):", cypher_results)
    print("\nSQL Results (PostgreSQL):", sql_results)

compare_queries()


# In[24]:


#Comparison of KNN query in Cypher and SQL model
#Reference input
longitude = 77.1025 
latitude = 28.7041  
k = 10

def execute_cypher_knn_query(long, lat, k):
    try:
        start_time = time.time()
        
        graph = Graph(uri="bolt://localhost:7687", auth=("neo4j", "ludhiana43"))

        cypher_query = """
        MATCH (f:Feature)
        WITH f, point.distance(point({longitude: $long, latitude: $lat}), f.coordinates) AS dist
        RETURN f.name AS feature, f.coordinates AS coordinates, dist
        ORDER BY dist ASC
        LIMIT $k;
        """

        results = graph.run(cypher_query, long=long, lat=lat, k=k).data()

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Cypher KNN Query Execution Time (Neo4j): {execution_time:.4f} seconds")       
        return execution_time, results
    except Exception as e:
        print(f"Error executing Cypher KNN query: {e}")
        return None, []

def execute_sql_knn_query(long, lat, k):
    try:
        start_time = time.time()
        conn = psycopg2.connect(database="mydb", user="postgres", password="postgres", host="localhost", port="5432")

        sql_query = """
        WITH distance_query AS (
            SELECT 
                id,
                feature,
                latitude,
                longitude,
                6371 * acos(
                    cos(radians(28.7041)) * cos(radians(latitude)) * cos(radians(longitude) - radians(77.1025)) +
                    sin(radians(28.7041)) * sin(radians(latitude))
                ) AS distance_km
            FROM test
        )
        SELECT *
        FROM distance_query
        ORDER BY distance_km ASC
        LIMIT 5;
        """

        cursor = conn.cursor()
        cursor.execute(sql_query, (long, lat, k))
        results = cursor.fetchall()

        end_time = time.time()

        execution_time = end_time - start_time
        print(f"SQL KNN Query Execution Time (PostgreSQL): {execution_time:.4f} seconds")

        cursor.close()
        conn.close()

        return execution_time, results
    except Exception as e:
        print(f"Error executing SQL KNN query: {e}")
        return None, []

def compare_knn_queries():
    cypher_time, cypher_results = execute_cypher_knn_query(longitude, latitude, k)

    sql_time, sql_results = execute_sql_knn_query(longitude, latitude, k)

    queries = ['Cypher KNN (Neo4j)', 'SQL KNN (PostgreSQL)']
    times = [cypher_time, sql_time]

    plt.figure(figsize=(10, 6))
    plt.bar(queries, times, color=['blue', 'orange'])
    plt.ylabel('Execution Time (seconds)')
    plt.title('KNN Query Execution Time: Neo4j vs PostgreSQL')
    plt.show()

compare_knn_queries()


# # Using Haversine Formula

# In[25]:


import time
import matplotlib.pyplot as plt
from py2neo import Graph
import psycopg2

# KNN parameters
longitude = 77.1025  # Example longitude
latitude = 28.7041   # Example latitude
k = 5  # Number of nearest neighbors

# --- Neo4j Setup and KNN Query Execution ---
def execute_cypher_knn_query(long, lat, k):
    try:
        start_time = time.time()
        
        # Connect to Neo4j
        graph = Graph(uri="bolt://localhost:7687", auth=("neo4j", "ludhiana43"))

        # Define the Cypher KNN query using the Haversine formula
        cypher_query = """
        MATCH (f:Feature)
        WHERE f.coordinates IS NOT NULL
        WITH f,
        toFloat(split(f.coordinates, ', ')[0]) AS lat2,
        toFloat(split(f.coordinates, ', ')[1]) AS long2
        WITH f, 
        6371 * 2 * asin(sqrt(haversin(radians(lat2 - $lat)) + cos(radians($lat)) * cos(radians(lat2)) * haversin(radians(long2 - $long)))) AS dist
        RETURN f.name AS feature, f.coordinates AS coordinates, dist
        ORDER BY dist ASC
        LIMIT $k;
        """

        # Execute the Cypher query
        results = graph.run(cypher_query, long=long, lat=lat, k=k).data()

        end_time = time.time()

        # Calculate the execution time
        execution_time = end_time - start_time
        print(f"Cypher KNN Query Execution Time (Neo4j): {execution_time:.4f} seconds")
        
        return execution_time, results
    except Exception as e:
        print(f"Error executing Cypher KNN query: {e}")
        return None, []

# --- PostgreSQL Setup and KNN Query Execution ---
def execute_sql_knn_query(long, lat, k):
    try:
        start_time = time.time()
        # Connect to PostgreSQL
        conn = psycopg2.connect(database="mydb", user="postgres", password="postgres", host="localhost", port="5432")

        # Define the SQL KNN query using the Haversine formula
        sql_query = f"""
        WITH distance_query AS (
            SELECT 
                id,
                feature,
                latitude,
                longitude,
                6371 * acos(
                    cos(radians({lat})) * cos(radians(latitude)) * cos(radians(longitude) - radians({long})) +
                    sin(radians({lat})) * sin(radians(latitude))
                ) AS distance_km
            FROM test
        )
        SELECT *
        FROM distance_query
        ORDER BY distance_km ASC
        LIMIT {k};
        """

        # Execute the SQL query
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()

        end_time = time.time()

        # Calculate the execution time
        execution_time = end_time - start_time
        print(f"SQL KNN Query Execution Time (PostgreSQL): {execution_time:.4f} seconds")

        cursor.close()
        conn.close()

        return execution_time, results
    except Exception as e:
        print(f"Error executing SQL KNN query: {e}")
        return None, []

# --- Comparison and Visualization ---
def compare_knn_queries():
    # Execute KNN query in Neo4j
    cypher_time, cypher_results = execute_cypher_knn_query(longitude, latitude, k)

    # Execute KNN query in PostgreSQL
    sql_time, sql_results = execute_sql_knn_query(longitude, latitude, k)

    # Plotting the comparison
    queries = ['Cypher KNN (Neo4j)', 'SQL KNN (PostgreSQL)']
    times = [cypher_time, sql_time]

    plt.figure(figsize=(10, 6))
    plt.bar(queries, times, color=['blue', 'orange'])
    plt.ylabel('Execution Time (seconds)')
    plt.title('KNN Query Execution Time: Neo4j vs PostgreSQL')
    plt.show()

compare_knn_queries()


# In[ ]:




