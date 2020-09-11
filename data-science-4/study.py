import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

countries = pd.read_csv("./data/countries.csv")

new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

# q1
string_columns_to_float = ['Agriculture', 'Arable', 'Birthrate', 'Climate',
                           'Coastline_ratio', 'Crops', 'Deathrate', 'Industry',
                           'Infant_mortality', 'Literacy', 'Net_migration', 'Other',
                           'Phones_per_1000', 'Pop_density', 'Service']
for column in string_columns_to_float:
    countries[column] = pd.to_numeric(countries[column].str.replace(',', '.'), errors='coerce')
countries['Region'] = countries['Region'].str.strip()
countries['Country'] = countries['Country'].str.strip()


def q2():
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    X = np.array(countries['Pop_density']).reshape(-1,1)
    y = countries['Region']
    result = est.fit_transform(X, y) >= 9.
    int(np.sum(result))


def q2_cool():
    pop_density = np.array(countries['Pop_density'])
    enc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    countries['Pop_dens_binned'] = enc.fit_transform(pop_density.reshape(-1,1))
    return len(countries.loc[countries['Pop_dens_binned'] >= 9.,'Country'])


def q3():
    enc = OneHotEncoder(handle_unknown='ignore')
    new_dt = pd.DataFrame(countries[['Region', 'Climate']])
    new_dt.Climate.fillna(0, inplace=True)
    result = enc.fit_transform(new_dt)
    print(result)


def q4():
    test_country = [
        'Test Country', 'NEAR EAST', -0.19032480757326514,
        -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
        0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
        0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
        -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
        -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
        0.263445277972641, 0.5712416961268142
    ]

    columns = countries.select_dtypes(include=['int64', 'float64']).columns

    pipeline = Pipeline(steps=[('imp', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler())])
    transformer = ColumnTransformer(transformers=[('number', pipeline, columns)], n_jobs=-1)

    transformer.fit(countries)

    test_dt = pd.DataFrame([test_country], columns=countries.columns)
    result = transformer.transform(test_dt)[0][columns.get_loc('Arable')]
    return round(float(result), 3)


def q5():
    q1, q3 = countries['Net_migration'].dropna().quantile([.25, .75])
    iqr = q3 - q1

    inferior_outlier = q1 - 1.5 * iqr
    upper_outlier = q3 + 1.5 * iqr

    outliers_below = int(countries[countries['Net_migration'] < inferior_outlier].shape[0])
    outliers_above = int(countries[countries['Net_migration'] > upper_outlier].shape[0])

    return outliers_below, outliers_above, False


def q6():
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=12)
    vectorizer = CountVectorizer()
    document_term_matrix = vectorizer.fit_transform(newsgroup.data)
    map_vocabulary = vectorizer.vocabulary_
    return int(document_term_matrix[:, map_vocabulary['phone']].sum())


def q7():
    tfidf_vectorizer = TfidfVectorizer().fit(newsgroup.data)
    tfidf_count = tfidf_vectorizer.transform(newsgroup.data)
    map_vocabulary = tfidf_vectorizer.vocabulary_
    return float(tfidf_count[:, map_vocabulary['phone']].sum().round(3))


q6()

