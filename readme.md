# song success predictor 

The objective is to predict the success of a music track using acoustic (tempo, key) and metadata (genre, song duration) features of the track.  
The measure of success is tied to a song's "hotttness" score.  This metric is assigned to tracks by the API providers based on mentions in news, play counts, radio airtime, Billboard rankings, and reviews on popular music websites.  This measure is directly correlated with the revenue of the track on market.  

There is heavy focus on ontological modeling, feature engineering, and model selection.  
 

Ontology describing feature space   
![music ontology](/images/07-ontology.jpg)



## Findings

+ metadata features are stronger indicators of hottt than acoustic 

* Acoustic features are poor indicators of hottt, but features derived from the raw acoustic features have more predictive power 
    * We decide that some of the acoustic features could  be combined into `energy` and `danceability`.
        - Find out that ontologies represent these measures as derived values from other features: 
            + `energy`: function of (loudness, segment stuff)
            + `danceability`: function of (tempo, time_signature)

+ Combination of a couple of diverse features does better
    * Combination of different energy calculations
    * Combination of different metadata features
    * Combination of different acoustic features  

+ The raw acoustic features perform fine on the training set
    * they actually perform better than the energy measures on the training set
    + energy measures **generalize better**.  theyre better on the test set



## To run 

Make sure the following files/folders are in the same directory:
+ tutorials/
+ MSongsDB/
+ MillionSongSubset/
+ swagmaster.db
+ create_track_metadata_db_custom.py


## master plan

1. [x] write script to build sample dataset 

2. [x] build another structure (pandas DataFrame?) to hold relevant fields for learning  

3. [ ] try to predict song_hotttnesss using other features 
    + acoustic 
        * key                 int,
        * tempo               real, 
        * loudness            real, 
        * time_signature      int, 
    + metadata 
        * duration            real, 
        * artist_familiarity  real,
        * artist_hotttnesss   real,
    + What learning models should we try?
        * Logistic regression
        * SVM
        * kNN


## building our dataset 

```sql
CREATE TABLE songs (
    track_id            text PRIMARY KEY,
    title               text,
    song_id             text,
    release             text,
    artist_id           text,
    artist_mbid         text,
    artist_name         text,
    duration            real,
    artist_familiarity  real,
    artist_hotttnesss   real,
    year                int,
    track_7digitalid    int,
    shs_perf            int,  # ???
    shs_work            int   # ???
    # new ones vvv
    song_hotttnesss     real, 
    danceability        real, 
    energy              real, 
    key                 int,
    tempo               real, 
    loudness            real, 
    time_signature      int
);
```


## Energy 
`energy`:  The feature mix we use to compute energy includes loudness and segment durations.




## Danceability 
`danceability`: We use a mix of features to compute danceability, including beat strength, tempo stability, overall tempo, and more.


## Notes

### Tutorial notebooks 
[MSD link to tutorials](https://labrosa.ee.columbia.edu/millionsong/pages/tutorial)
#### tutorial_1
+ Shows how to iterate over the files within the MillionSongSubset
+ The AdditionFiles has sql databases set up to ping into the /data folder's contents 
+ Runs through an exercise to find out which artist has the most songs in the dataset (by artist_id)


#### tutorial_3_track-metadata 
+ Shows how to interface with the dataset (in db form) using sqlite.
    * There are .db files in AdditionalFiles.  This one uses track_metadata (subset_track_metadata.db)

+ `subset_track_metadata.db`
    * Contains one table named 'songs'
    * Contains the following columns
        - track_id text PRIMARY KEY, 
        - title text, 
        - song_id text, 
        - release text, 
        - artist_id text, 
        - artist_mbid text, 
        - artist_name text, 
        - duration real, 
        - artist_familiarity real, 
        - artist_hotttnesss real, 
        - year int
+ Some useful queries:
    * Get all songs without MB ID's : `SELECT artist_id,artist_mbid FROM songs WHERE artist_mbid=''`
    * Get all distinct artists: `SELECT DISTINCT artist_id, artist_name FROM songs`
    * Get all dudes with a float>value: `SELECT DISTINCT artist_name, artist_familiarity FROM songs WHERE artist_familiarity>.8`
        - Can use this one to filter out the tracks where hotttnesss is 0. (empty data) (WHERE NOT artist_hotttnesss=0)

## References 

+ [github link -- matches artist names from yahoo ratings set to the MSD (by artist_id)](https://github.com/tbertinmahieux/MSongsDB/blob/master/Tasks_Demos/YahooRatings/match_artist_names.py)

+ [gitlub link -- MSongDB code.  Has python scripts and examples and stuff](https://github.com/tbertinmahieux/MSongsDB)
  
+ [github link -- homeboi kevin over here did all the tutorials in python notebooks](https://github.com/kevin11hg/msong)
  
+ [google group forum thing](https://groups.google.com/forum/#!forum/millionsongdataset)
  
+ [jupnbk. project that shows how to create subset datasets](http://nbviewer.jupyter.org/github/ds3-at-ucsd/msd-fp-p1/blob/master/grab_msd_data.ipynb)
  
+ [github link --- some project that uses spark](https://github.com/hsudarshan/Trend_Analysis_MSD_using_Spark/blob/master/CSE740ProjectReport.pdf)
  
+ [year predictive modeling](http://ds3-at-ucsd.github.io/msd-fp-p1/)
