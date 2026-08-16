from conftest import CATEGORIES


def test_tfidf_and_model_agree_on_feature_count(app_module):
    n_features = len(app_module.tfidf.get_feature_names_out())
    assert app_module.model.n_features_in_ == n_features


def test_model_classes_are_the_expected_categories(app_module):
    assert set(app_module.model.classes_) == CATEGORIES


def test_vectorizer_uses_word_1_2_grams(app_module):
    assert app_module.tfidf.ngram_range == (1, 2)


def test_pipeline_end_to_end(app_module):
    cleaned = app_module.clean_text("SWIGGY ORDER INR 350 TXNaa11bb22")
    vector = app_module.tfidf.transform([cleaned])
    assert app_module.model.predict(vector)[0] in CATEGORIES
