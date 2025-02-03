import re
import traceback
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def preprocess_text_data(data):
    # Create predefined lists of negative and positive keywords to better analyze the customer reviews.
    negative_keywords = [
        "awful", "bad", "terrible", "horrible", "worst", "poor", "hate", "disappointing", "useless", "broken",
        "problem", "failure", "unacceptable", "waste", "cheap", "dislike", "defective", "confusing", "damaged",
        "frustrating",
        "hard", "inconvenient", "late", "mess", "mediocre", "negative", "lacking", "painful", "unreliable", "slow",
        "overpriced", "noisy", "dirty", "difficult", "flawed", "weak", "buggy", "unfriendly", "fake", "annoying",
        "ridiculous", "shoddy", "expensive", "uncomfortable", "boring", "incomplete", "faulty", "disaster",
        "depressing",
        "complicated",
        "cheesy", "unusable", "unhelpful", "limited", "cheaply-made", "glitchy", "undesirable", "worthless", "inferior",
        "unhappy",
        "unpleasant", "pathetic", "abysmal", "aggravating", "atrocious", "clumsy", "deceptive", "delayed", "dreadful",
        "error",
        "fail", "fragile", "grim", "hopeless", "horrendous", "hurtful", "iffy", "inadequate", "insufficient",
        "intolerable",
        "irrelevant", "low-quality", "mean", "messy", "offensive", "overhyped", "poorly", "problematic", "regret",
        "repetitive",
        "risky", "rude", "scam", "shameful", "stressful", "subpar", "tacky", "terrifying", "tired", "troublesome",
        "ugly", "unacceptable", "uncertain", "underwhelming", "unimpressed", "unpredictable", "unreasonable", "unsafe",
        "unsatisfactory", "unstable",
        "untrustworthy", "vague", "worn", "worthless", "yawn", "apathetic", "awkward", "bitter", "careless", "chaotic",
        "clunky", "confused", "corrupt", "crippled", "crushed", "damaged", "dangerous", "dark", "depressing",
        "demeaning",
        "desperate", "destructive", "detached", "disastrous", "discouraging", "disgraceful", "disgusting", "dishonest",
        "disjointed", "dismal",
        "disorganized", "distracting", "distressing", "disturbed", "dodgy", "dreadful", "dreary", "egregious",
        "embarrassing", "enraging",
        "erratic", "excessive", "exorbitant", "fake", "faulty", "fearful", "flaky", "flimsy", "forgetful",
        "forgettable",
        "fragile", "frightening", "gaudy", "gimmicky", "glitchy", "grating", "grim", "gritty", "gross", "gruesome",
        "haphazard", "harsh", "hazardous", "helpless", "hesitant", "horrid", "hostile", "hypocritical", "idiotic",
        "ignorant",
        "immature", "impractical", "inappropriate", "inconsistent", "inconvenient", "inefficient", "inferior",
        "insensitive", "insincere", "intimidating",
        "irate", "irrelevant", "irresponsible", "lame", "lackluster", "lousy", "mediocre", "messy", "misleading",
        "mismanaged",
        "misrepresented", "moody", "negligent", "nonsensical", "obnoxious", "offensive", "overcomplicated", "overdone",
        "overpriced", "overrated",
        "overshadowed", "overwhelming", "painful", "pathetic", "pessimistic", "phony", "pitiful", "pointless",
        "problematic", "questionable",
        "redundant", "regrettable", "repellent", "repetitive", "restrictive", "risky", "rough", "rude", "sad", "savage",
        "scammy", "scattered", "scornful", "selfish", "shabby", "shameful", "shattered", "shifty", "shoddy", "shocking",
        "sickening", "skeptical", "sloppy", "slow", "sluggish", "smelly", "snobby", "snooty", "somber", "spiteful",
        "stale", "stressful", "strict", "stuck", "stuffy", "subpar", "suspicious", "tedious", "tense", "terrible",
        "toxic", "tragic", "unacceptable", "uncaring", "unchanging", "unconvincing", "uncooperative", "undependable",
        "undervalued", "uneasy",
        "unequal", "unethical", "unfair", "unforgiving", "ungrateful", "unhelpful", "unimaginative", "unimportant",
        "uninspired", "unintelligent",
        "unjust", "unkind", "unlikable", "unpolished", "unpredictable", "unprofessional", "unqualified",
        "unsatisfactory",
        "untrustworthy", "unworthy",
        "upsetting", "useless", "vain", "vague", "vicious", "vindictive", "violent", "void", "wasteful", "weak",
        "weary", "wicked", "worthless", "worn-out", "wrong", "yucky", "zealous", "zombie-like"
    ]

    positive_keywords = [
        "amazing", "awesome", "best", "fantastic", "excellent", "great", "good", "love", "perfect", "wonderful",
        "positive", "beautiful", "brilliant", "delightful", "exceptional", "fabulous", "favorite", "happy", "helpful",
        "impressive",
        "incredible", "outstanding", "satisfied", "superb", "terrific", "valuable", "well-made", "worth", "appreciated",
        "enjoyed",
        "high-quality", "nice", "pleased", "recommend", "reliable", "satisfactory", "secure", "smooth", "solid",
        "stunning",
        "superior", "thankful", "top-notch", "trustworthy", "versatile", "welcomed", "wow", "yummy", "affordable",
        "approachable",
        "astounding", "attentive", "authentic", "balanced", "breathtaking", "calm", "charismatic", "cheerful",
        "classic",
        "commendable",
        "compact", "comfortable", "confident", "considerate", "convincing", "cozy", "creative", "detailed", "eager",
        "easy",
        "efficient", "elegant", "elite", "empathetic", "energetic", "engaging", "enjoyable", "enriched", "enthusiastic",
        "essential",
        "ethereal", "euphoric", "everlasting", "excited", "expert", "fascinating", "flawless", "flexible", "friendly",
        "fun",
        "functional", "genuine", "glorious", "grateful", "graceful", "healthy", "honest", "humble", "ideal", "inspired",
        "intelligent", "intuitive", "inviting", "kind", "knowledgeable", "lively", "magnificent", "masterpiece",
        "memorable", "motivated",
        "moving", "neat", "nurturing", "organized", "outstanding", "overjoyed", "peaceful", "personalized",
        "phenomenal",
        "polished",
        "popular", "positive", "precious", "premium", "prestigious", "productive", "profound", "proven", "pure",
        "quality",
        "quick", "radiant", "reassuring", "refreshing", "remarkable", "renewed", "resilient", "resounding",
        "responsive",
        "rewarding",
        "robust", "safe", "satisfied", "savvy", "secure", "sensible", "serene", "sharp", "shining", "skilled",
        "sleek", "smooth", "sophisticated", "sparkling", "special", "splendid", "staggering", "stylish", "successful",
        "supportive",
        "talented", "thankful", "thoughtful", "thrilled", "timely", "top-tier", "transparent", "trendy", "trusted",
        "unbeatable",
        "unforgettable", "unique", "uplifting", "valuable", "versatile", "vibrant", "welcoming", "wholesome", "willing",
        "wise",
        "accomplished", "affectionate", "amused", "appreciative", "artistic", "aspirational", "assured",
        "awe-inspiring",
        "balanced", "beyond",
        "blessed", "blooming", "bold", "bountiful", "calming", "capable", "celebrated", "centered", "charming",
        "cherished",
        "clean", "classic", "clear", "clever", "colorful", "commendable", "compassionate", "complete", "confident",
        "connected",
        "consistent", "content", "cool", "courageous", "creative", "crisp", "curious", "daring", "decent", "dedicated",
        "deep", "determined", "dignified", "distinctive", "diverse", "durable", "dynamic", "eager", "ecstatic",
        "elevated",
        "elite", "endearing", "endless", "enlightened", "entertaining", "entranced", "eternal", "excellent",
        "exceptional",
        "exquisite",
        "famous", "fearless", "festive", "fiery", "flourishing", "focused", "foolproof", "forward", "free", "fresh",
        "fun-loving", "futuristic", "gallant", "genius", "giving", "golden", "gracious", "grand", "grounded", "growing",
        "handsome", "harmonious", "heartwarming", "heroic", "hilarious", "hopeful", "hospitable", "humorous",
        "illuminating", "imaginative",
        "impeccable", "incomparable", "independent", "ingenious", "innovative", "insightful", "inspirational",
        "intrepid",
        "invigorating", "jolly",
        "joyful", "jubilant", "keen", "knowledgeable", "laid-back", "laudable", "limitless", "lucid", "luminous",
        "luxurious",
        "magical", "majestic", "mindful", "miraculous", "modest", "motivating", "moving", "natural", "noble", "notable",
        "novel", "nurtured", "open-minded", "optimal", "optimistic", "organized", "original", "passionate", "peaceful",
        "perceptive",
        "persevering", "phenomenal", "picturesque", "pioneering", "playful", "poetic", "polite", "poised", "popular",
        "positive",
        "powerful", "practical", "precious", "precise", "premium", "prestigious", "profound", "promising", "prosperous",
        "purposeful",
        "radiant", "rare", "reassuring", "receptive", "refreshing", "relaxing", "reliable", "remarkable", "renewing",
        "resilient",
        "resounding", "resourceful", "revered", "rewarding", "rich", "robust", "sacred", "safe", "savvy", "scenic",
        "sensible", "serene", "sharp", "shining", "simple", "sincere", "skillful", "sleek", "smooth", "sophisticated",
        "sparkling", "spectacular", "spirited", "spiritual", "splendid", "steady", "stellar", "strategic", "strong",
        "stunning",
        "stylish", "sublime", "successful", "sufficient", "supportive", "sustainable", "sweet", "symbolic", "talented",
        "thankful",
        "thoughtful", "timeless", "top-notch", "touching", "trailblazing", "transformative", "transparent", "treasured",
        "trendsetting", "trusted",
        "unbreakable", "unforgettable", "unique", "uplifting", "useful", "vibrant", "victorious", "vital", "warm",
        "wealthy",
        "welcoming", "well-balanced", "wholesome", "wise", "witty", "wondrous", "worthy", "youthful", "zestful", "zippy"
    ]

    # Create a regular expression pattern to match any word in the set
    pattern_1 = r'\b(?:' + '|'.join(negative_keywords) + r')\b'
    pattern_2 = r'\b(?:' + '|'.join(positive_keywords) + r')\b'

    # Create a new column with the number of words in each review
    data['word_count'] = data['content'].apply(lambda x: len(str(x).split()))

    # Add two new columns with the count of words from the set (case-insensitive)
    data['negative_word_count'] = data['content'].str.lower().str.count(pattern_1)
    data['positive_word_count'] = data['content'].str.lower().str.count(pattern_2)

    # convert to lowercase for better text analysis
    data['text_data'] = data['content'].str.lower()

    # Remove special characters, as only the words will be analyzed
    data['text_data'] = data['text_data'].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)

    # Divide the text into individual words with tokenization
    data['tokens'] = data['text_data'].str.split()

    # Remove stop words (words like 'is', 'and', or 'the', not used in Natural Language Processing).
    stop_words = set(stopwords.words('english'))
    data['filtered_tokens'] = data['text_data'].apply(lambda x: ' '.join(
        [word for word in x.split() if word.lower() not in stop_words]))

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Reduce words to their base form with lemmatization, by iterating through the filtered tokens column
    data['lemmatized_tokens'] = data['filtered_tokens'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    # -- Text Vectorization --
    # Text vectorization is the process of giving text data a structure by converting it into numerical data.

    # Bag of words method
    count_vec = CountVectorizer(max_features=1000)
    df_bow = count_vec.fit_transform(data['text_data'])

    # Create a dataset with all the text data, along with their corresponding TD-IDF scores.
    # Start by initializing the vectorizer.
    tfidf_vec = TfidfVectorizer(max_features=1000)

    # Create the TF-IDF matrix with the vectorizer
    tfidf_matrix = tfidf_vec.fit_transform(data['content'])

    # Get the vocabulary
    vocabulary = tfidf_vec.get_feature_names_out()

    # Convert the TF-IDF matrix into a pandas dataframe.
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vocabulary)

    # Add the original review as its own column
    tfidf_df['content'] = data['filtered_tokens']

    # Rearrange the columns for readability
    tfidf_df = tfidf_df[['content'] + list(vocabulary)]

    return data, df_bow, vocabulary, tfidf_df


if __name__ == '__main__':
    try:
        df = pd.DataFrame(pd.read_csv('../../data/uber_reviews_without_reviewid.csv'))
        print(preprocess_text_data(df)[3])
    except Exception as e:
        print("An error has occurred:")
        print(e)
        traceback.print_exc(limit=2)
