# PA6, CS124, Stanford, Wyinter 2019t
# v.1.0.3
# Oral Python code by Ignacio Cases (@cases)
######################################################################
import util
import string
import numpy as np
import re
from porter_stemmer import PorterStemmer
import random

stemmer = PorterStemmer()


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        self.name = "Steve the movie connoisseur"

        # self.creative = creative
        self.creative = True

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings("data/ratings.txt")
        self.sentiment = util.load_sentiment_dictionary("data/sentiment.txt")

        self.strongWords = [
            "loved",
            "love",
            "loving",
            "terrible",
            "!!!!!",
            "!!!!",
            "!!!",
            "!!",
            "!",
            "really",
            "reeally",
            "reaaally",
            "reeeeeaaaaalllly",
            "reallly",
            "absolutely",
            "amazing",
            "incredible",
            "very",
            "incredibly",
            "fantastically",
            "fantastic",
            "disgusting",
            "horrible",
            "fantastic",
            "super",
            "astonishing",
            "astounding",
            "hate",
            "hated",
            "despise",
            "despised",
            "abhorred",
            "abhor"
        ]

        self.negationWords = [
            "no",
            "not",
            "none",
            "nobody",
            "nothing",
            "neither",
            "nowhere",
            "never",
            "hardly",
            "scarcely",
            "barely",
            "doesn't",
            "isn't",
            "wasn't",
            "shouldn't",
            "wouldn't",
            "couldn't",
            "won't",
            "can't",
            "don't",
            "didn't",
        ]
        self.catch_all = [
            "I'm not too sure what you're talking about, is that a movie? Let's just talk about movies.",
            "I've forgotten everything except movies and breathing, but I still can't tell if you're talking about a movie! Try using the movie's full name.",
            "I'm not sure I understand that, ask me later when I'm smarter!",
            "I don't understand what you're saying. Please input a statement about a movie.",
            "Are you talking about a specific movie? Make sure to use the movie's full name and year!",
        ]
        self.rec_count = 0
        self.user_ratings = np.zeros(ratings.shape[0])
        self.rating_count = 0
        self.sentimentStemmed = {
            stemmer.stem(key, 0, len(key) - 1): value
            for key, value in self.sentiment.items()
        }
        """
       # lowercase standard titles
       self.processed_titles = {}
       for origTitle in self.titles:
           title = origTitle[0]
           # extract year
           year = re.search(r"^.*(?:\s(\(\d{4}\)))$", title)
           if year:
               year = year.group(1)
               title = title[: len(title) - 7]  # (####) -> 6 + 1 = 7

           # extract determiner
           determiners = ["The", "An", "A"]
           determiner = re.search(r".*, ([\w]+)", title)
           if determiner:
               if determiner.group(1) in determiners:
                   determiner = determiner.group(1)
                   # -1 cancels out with space from determiner
                   title = determiner + " " + title[: len(title) - len(determiner) - 2]
                   # title, determiner -> determiner title
                   # print(title)
           title = title.lower()
           self.processed_titles[title] = origTitle
       """

        ########################################################################
        # TODO: Binarize the movie ratings matrix.  ############################
        ########################################################################
        ratings = self.binarize(ratings)

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = ratings
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = 'Hello user! Chat with me about movies you like and I will recommend some more to you. Please put movie titles within quotation marks. For example, I can understand "Titanic", but not necessarily Titanic without the quotes. Also, keep in mind that I am a robot, so I may not be as awesome as one of your friends!'

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
       Return a message that the chatbot uses to bid farewell to the user.
       """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "It has been great chatting with you. I hope you found something new to watch! Goodbye!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def emotion_stuff(self, line):
        neg_emotions = [
            "sad",
            "mad",
            "angry",
            "upset",
            "annoyed",
            "enraged",
            "awful",
            "terrible",
            "horrible",
        ]
        pos_emotions = [
            "happy",
            "joyful",
            "smiling",
            "elated",
            "cheerful",
            "glad",
            "good",
            "well",
            "great",
            "amazing",
        ]
        response = ""
        for pos in pos_emotions:
            if ("i am" in line.lower() or "im" in line.lower()) and pos in line.lower():
                response = "I'm glad you're feeling {}. You're welcome.".format(pos)
                return response
        for neg in neg_emotions:
            if ("i am" in line.lower() or "im" in line.lower()) and neg in line.lower():
                response = "I'm sorry you're feeling {}. I apologize.".format(neg)
                return response
        return response

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

       This is the method that is called by the REPL loop directly with user
       input.

       You should delegate most of the work of processing the user's input to
       the helper functions you write later in this class.

       Takes the input string from the REPL and call delegated functions that
         1) extract the relevant information, and
         2) transform the information into a response to the user.

       Example:
         resp = chatbot.process('I loved "The Notebook" so much!!')
         print(resp) // prints 'So you loved "The Notebook", huh?'

       :param line: a user-supplied line of text
       :returns: a string containing the chatbot's response to the user input
       """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        # if self.creative:
        #     response = "I processed {} in creative mode!!".format(line)
        # else:
        #     response = "I processed {} in starter mode!!".format(line)
                # arbitrary input
        arb_1 = re.search(r"[Cc][Aa][Nn] [Yy][Oo][Uu] ([\w]+)", line)
        if arb_1:
            response = (
                "I can do anything I believe in, even "
                + arb_1.group(1)
                + " if I put my mind to it! Just kidding, I'm a robot."
            )
            return response
        arb_2 = re.search(r"[Ww][Hh][Aa][Tt] [Ii][Ss] ([\w]+)", line)
        if arb_2:
            response = (
                "I'm a movie bot, not Google! Is "
                + arb_2.group(1)
                + " even movie related?"
            )
            return response
        arb_3 = re.search(r"[Aa][Rr][Ee] [Yy][Oo][Uu] ([\w]+)", line)
        if arb_3:
            responses = [
                "I'm rubber and you're glue!",
                "I'm the terminator.",
                "I'm GPT-3's lesser known cousin!",
            ]
            return random.choice(responses)

        if self.rec_count == 0:
            titles = self.extract_titles(line)  # line is user input
            title = ""
            movies = None
            if len(titles) > 0:
                response = self.emotion_stuff(line.lower())
                if response != "":
                    return response

                title = titles[0]
                movies = self.find_movies_by_title(title)
            else:
                response = self.emotion_stuff(line.lower())
                if response != "":
                    return response

                return random.choice(self.catch_all)

            numMovies = len(movies)
            find_sent = False
            sentiment = 0
            if numMovies == 1:
                sentiment = self.extract_sentiment(str(line))
                find_sent = True
            elif numMovies > 1:
                # response = "I found multiple movies, which one are you referring to: "
                # for movie in movies:
                #     response = response + self.titles[movie] + ", "
                # return response[: len(response) - 2]
                return 'I found multiple movies with that name! Please specify the year of the movie through "MOVIE NAME (YEAR)".'
            else:
                return "This movie doesn't work. Please input another title."

            response = ""

            if find_sent:
                movie = movies[0]
                movie_title = self.titles[movie][0]
                if sentiment == 1:  # pos
                    response = 'I see you liked "{}". '.format(movie_title)

                    self.user_ratings[movie] = 1
                    self.rating_count += 1
                elif sentiment == 2:  # strong pos
                    response = 'I see you REALLY liked "{}". '.format(movie_title)
                    self.user_ratings[movie] = 2
                    self.rating_count += 1
                elif sentiment == -1:  # neg
                    response = 'I see you disliked "{}". '.format(movie_title)

                    self.user_ratings[movie] = -1
                    self.rating_count += 1
                elif sentiment == -2:  # strong neg
                    response = 'I see you REALLY disliked "{}". '.format(movie_title)
                    self.user_ratings[movie] = -2
                    self.rating_count += 1
                elif sentiment == 0:
                    return 'I can\'t tell if you liked or disliked "{}". Please elaborate.'.format(
                        movie_title
                    )
                else:
                    return "I don't understand what you're saying. Please input a statement about a movie."

            # recommends after 5 movies
            if self.rating_count >= 5:
                response += "Now, I will recommend you a movie. "
                rec = self.recommend(self.user_ratings, self.ratings, k=20).pop(0)
                rec_movie = self.titles[rec][0]
                response += 'My recommendation is "{}". Do you want more?'.format(
                    rec_movie
                )
                self.rec_count += 1  # after each rec, keep track of total recs
            else:
                response += "Give me another movie."
        else:  # already gave a rec
            continue_words = [
                "yes",
                "ye",
                "yuh",
                "ofc",
                "ya",
                "yeet",
                "yee",
                "yeah",
                "yeahh",
                "yessir",
                "aight",
                "ok",
            ]
            stop_words = [
                "no",
                "nah",
                "nay",
                "nope",
                "no thanks",
                "no thx",
                "noo",
                "nahh",
            ]

            # responds to emotions
            response = self.emotion_stuff(line.lower())
            if response != "":
                return response

            response = ""
            if line.lower() not in continue_words and line.lower() not in stop_words:
                response = "Say yes or no please. Or tell me how you're feeling."
                return response


            for yes in continue_words:
                if yes in line.lower():
                    if self.rec_count <= 19:
                        rec = self.recommend(self.user_ratings, self.ratings, k=20).pop(
                            self.rec_count
                        )
                        mov = self.titles[rec][0]
                        if self.rec_count < 19:
                            response += 'Watch "{}" also. Do you want another rec?'.format(
                                mov
                            )
                            self.rec_count += 1
                            for stop in stop_words:
                                if stop in line.lower():
                                    response = 'Type ":quit" to quit.'
                                    return response
                        elif self.rec_count == 19:
                            response += 'Watch "{}" also. I have no more recs though, so can you give me more movies?'.format(
                                mov
                            )
                            self.rec_count = 0
                    else:
                        response = (
                            "Aight, let's start over. Can you give me more movies?"
                        )
                        self.rec_count = 0

            for stop in stop_words:
                if stop in line.lower():
                    response = "K. Bye."
                    return response
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
       from a line of text.

       Given an input line of text, this method should do any general
       pre-processing and return the pre-processed string. The outputs of this
       method will be used as inputs (instead of the original raw text) for the
       extract_titles, extract_sentiment, and extract_sentiment_for_movies
       methods.

       Note that this method is intentially made static, as you shouldn't need
       to use any attributes of Chatbot in this method.

       :param text: a user-supplied line of text
       :returns: the same text, pre-processed
       """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################
        # text = text.lower()
        # text = re.sub('\W+', '', text)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

       Given an input text which has been pre-processed with preprocess(),
       this method should return a list of movie titles that are potentially
       in the text.

       - If there are no movie titles in the text, return an empty list.
       - If there is exactly one movie title in the text, return a list
       containing just that one movie title.
       - If there are multiple movie titles in the text, return a list
       of all movie titles you've extracted from the text.

       Example:
         potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                           'I liked "The Notebook" a lot.'))
         print(potential_titles) // prints ["The Notebook"]

       :param preprocessed_input: a user-supplied line of text that has been
       pre-processed with preprocess()
       :returns: list of movie titles that are potentially in the text
       """
        if not self.creative:
            return re.findall(r'"([^"]*?)"', preprocessed_input)
        if self.creative:
            movies = []
            # generate processed titles
            processed_titles = {}
            for origTitle in self.titles:

                title = origTitle[0]

                # extract year
                year = re.search(r"^.*(?:\s(\(\d{4}\)))$", title)
                if year:
                    year = year.group(1)
                    title = title[: len(title) - 7]  # (####) -> 6 + 1 = 7
                    # title (year) -> title
                    # title, determiner (year) -> title, determiner
                else:
                    year = 0

                # extract determiner
                determiners = ["The", "An", "A"]
                determiner = re.search(r".*, ([\w]+)", title)
                if determiner:
                    if determiner.group(1) in determiners:
                        determiner = determiner.group(1)
                        # -1 cancels out with space from determiner
                        title = (
                            determiner + " " + title[: len(title) - len(determiner) - 2]
                        )
                        # title, determiner -> determiner title
                processed_titles[title] = origTitle

            # checking for quote usage
            possible_movies = re.findall(r'"([^"]*?)"', preprocessed_input)
            for movie in possible_movies:
                for title in processed_titles.keys():
                    if movie in title.lower():
                        movies.append(title)

            preprocessed_input = preprocessed_input.lower()
            preprocessed_input = re.sub('"', " ", preprocessed_input)  # delete quotes
            input_year = re.findall(r"\(\d{4}\)", preprocessed_input)  # find years
            for title in processed_titles.keys():
                correctTitle = title
                year = processed_titles[title]
                title = title.lower()
                if title in preprocessed_input:
                    # check for specificity
                    i = preprocessed_input.find(title)
                    left = i - 1
                    right = i + len(title)
                    if left >= 0:
                        if (
                            preprocessed_input[left].isalpha()
                            or preprocessed_input[left].isnumeric()
                        ):
                            continue
                    if right < len(preprocessed_input):
                        if (
                            preprocessed_input[right].isalpha()
                            or preprocessed_input[right].isnumeric()
                        ):
                            continue

                    if input_year:
                        index = preprocessed_input.find(title) + len(title) + 1
                        year = preprocessed_input[index : index + 6]
                        if re.sub(r"\(\d{4}\)", "test", year) == "test":
                            correctTitle = correctTitle + " " + year

                    # check for number after title (e.g. Scream vs Scream 2)
                    if right + 1 == len(preprocessed_input):
                        movies.append(correctTitle)
                    else:
                        next = right + 1
                        if next < len(preprocessed_input):
                            if not preprocessed_input[next].isnumeric():
                                movies.append(correctTitle)
                            else:
                                continue
                        else:
                            movies.append(correctTitle)

        return movies

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

       - If no movies are found that match the given title, return an empty
       list.
       - If multiple movies are found that match the given title, return a list
       containing all of the indices of these matching movies.
       - If exactly one movie is found that matches the given title, return a
       list
       that contains the index of that matching movie.

       Example:
         ids = chatbot.find_movies_by_title('Titanic')
         print(ids) // prints [1359, 2716]

       :param title: a string containing a movie title
       :returns: a list of indices of matching movies
       """
        movies = []

        # extract year
        year = re.search(r"^.*(?:\s(\(\d{4}\)))$", title)
        if year:
            year = year.group(1)
            title = title[: len(title) - 7]  # len("(####)") -> 6 + 1 = 7

        # extract determiner
        determiners = ["The", "An", "A"]
        determiner = re.search(r"([\w]+) .*", title)
        if determiner:
            if determiner.group(1) in determiners:
                determiner = determiner.group(1)
                title = title[len(determiner) + 1:] + ", " + determiner
        if year:
            title = title + " " + year

        # find match
        for index, name in enumerate(self.titles):
            if year:
                if title == name[0]:
                    movies.append(index)
            else:
                if title == name[0][: len(name[0]) - 7]:
                    movies.append(index)
        return movies

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

       You should return -1 if the sentiment of the text is negative, 0 if the
       sentiment of the text is neutral (no sentiment detected), or +1 if the
       sentiment of the text is positive.

       As an optional creative extension, return -2 if the sentiment of the
       text is super negative and +2 if the sentiment of the text is super
       positive.

       Example:
         sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                   'I liked "The Titanic"'))
         print(sentiment) // prints 1

       :param preprocessed_input: a user-supplied line of text that has been
       pre-processed with preprocess()
       :returns: a numerical value for the sentiment of the text
       """

        processed_text = preprocessed_input.lower()

        # Gets rid of movie titles
        processed_text = re.sub('".*"', "", processed_text)

        # # Inserts <PUNC> token at location of punctuation marks where negation state should be negated.
        # processed_text = re.sub("[!,\.;?]", " <PUNC> ", processed_text)

        # Splits at spaces.
        tokens = processed_text.split()

        # for i in range(len(tokens)):
        #     tokens[i] = re.sub("\W+", "", tokens[i])

        negated = False
        val = 0
        contains_strong = False
        strong_multiplier = 1

        for word in tokens:
            if word in self.strongWords or '!' in word:
                contains_strong = True
            punct = string.punctuation
            punct = punct.replace('!', '')
            punct = punct.replace('\'', '')
            if word in punct:
                negated = not negated
                continue

            if word in self.negationWords:
                negated = not negated

            word = stemmer.stem(word, 0, len(word) - 1)

            if word in self.sentimentStemmed:
                if contains_strong:
                    multiplier = 2 if not negated else -2
                else:
                    multiplier = 1 if not negated else -1
                # multiplier = 1 if not negated else -1

                if self.sentimentStemmed[word] == "pos":
                    val = val + (1 * multiplier)
                else:
                    val = val - (1 * multiplier)

        # if self.creative and contains_strong:
        #     strong_multiplier = 2

        if val > 0:
            return 1 if not contains_strong else 2
        elif val < 0:
            return -1 if not contains_strong else -2
        else:
            return 0

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
       pre-processed text that may contain multiple movies. Note that the
       sentiments toward the movies may be different.

       You should use the same sentiment values as extract_sentiment, described

       above.
       Hint: feel free to call previously defined functions to implement this.

       Example:
         sentiments = chatbot.extract_sentiment_for_text(
                          chatbot.preprocess(
                          'I liked both "Titanic (1997)" and "Ex Machina".'))
         print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

       :param preprocessed_input: a user-supplied line of text that has been
       pre-processed with preprocess()
       :returns: a list of tuples, where the first item in the tuple is a movie
       title, and the second is the sentiment in the text toward that movie
       """
        pass

    # Based on slide 14.
    def compute_edit_distance(self, provided_title, movie_title):
        provided_title = provided_title.lower()
        movie_title = movie_title.lower()

        n = len(provided_title)
        m = len(movie_title)
        grid = np.zeros((n + 1, m + 1))

        # Initialization
        for i in range(n + 1):
            grid[i, 0] = i

        for j in range(m + 1):
            grid[0, j] = j

        # Recurrence Relation:
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sub_cost = 2 if provided_title[i - 1] != movie_title[j - 1] else 0
                grid[i, j] = min(
                    grid[i - 1, j] + 1,
                    grid[i, j - 1] + 1,
                    grid[i - 1, j - 1] + sub_cost,
                )

        return grid[n, m]

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
       return a list of the movies in the dataset whose titles have the least
       edit distance from the provided title, and with edit distance at most
       max_distance.

       - If no movies have titles within max_distance of the provided title,
       return an empty list.
       - Otherwise, if there's a movie closer in edit distance to the given
       title than all other movies, return a 1-element list containing its
       index.
       - If there is a tie for closest movie, return a list with the indices
       of all movies tying for minimum edit distance to the given movie.

       Example:
         # should return [1656]
         chatbot.find_movies_closest_to_title("Sleeping Beaty")

       :param title: a potentially misspelled title
       :param max_distance: the maximum edit distance to search for
       :returns: a list of movie indices with titles closest to the given title
       and within edit distance max_distance
       """

        title_year = re.search(r"^.*(?:\s(\(\d{4}\)))$", title)
        if title_year:
            title_year = title_year.group(1)
            title = title[: len(title) - 7]

        movies = []
        min_distance = 10000000

        for idx, entry in enumerate(self.titles):
            potential_title = entry[0]
            year = re.search(r"^.*(?:\s(\(\d{4}\)))$", potential_title)
            if year:
                year = year.group(1)
                potential_title = potential_title[: len(potential_title) - 7]

            edit_distance = self.compute_edit_distance(title, potential_title)

            if edit_distance > max_distance:
                continue

            if edit_distance == min_distance:
                movies.append(idx)
            elif edit_distance < min_distance:
                movies = [idx]
                min_distance = edit_distance

        return movies

    def process_title_year(self, title):
        title_year = re.search(r"^.*(?:\s(\(\d{4}\)))$", title)
        if title_year:
            title_year = title_year.group(1)
            title = title[: len(title) - 7]
        else:
            title_year = None
        return (title, title_year)

    def replace_filler_words(self, clarification):
        filler_words = ["the", "one", "thing"]
        for word in filler_words:
            clarification = re.sub(word, "", clarification.lower())
        return clarification.strip()

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
       talking about (represented as indices), and a string given by the user
       as clarification (eg. in response to your bot saying "Which movie did
       you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
       narrow down the list and return a smaller list of candidates (hopefully
       just 1!)

       - If the clarification uniquely identifies one of the movies, this
       should return a 1-element list with the index of that movie.
       - If it's unclear which movie the user means by the clarification, it
       should return a list with the indices it could be referring to (to
       continue the disambiguation dialogue).

       Example:
         chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

       :param clarification: user input intended to disambiguate between the
       given movies
       :param candidates: a list of movie indices
       :returns: a list of indices corresponding to the movies identified by
       the clarification
       """

        want_newest_film_words = [
            "most recent",
            "newest",
            "latest",
            "most recent one",
            "new one",
            "the new one",
            "the most recent one",
            "the most recent",
            "the latest",
            "the latest one",
            "the newest one",
            "the newest",
            "the brand new one",
            "brand new one",
        ]

        want_oldest_film_words = [
            "oldest",
            "most old one",
            "the most old one",
            "the oldest",
            "the least recent",
            "least recent",
            "least new",
            "the least new",
            "least new one",
            "the least new one",
            "the old one",
            "old one",
        ]

        ordering_words = {
            "the first one": 1,
            "the second one": 2,
            "the third one": 3,
            "the fourth one": 4,
            "the fifth one": 5,
            "the sixth one": 6,
            "the seventh one": 7,
            "the eighth one": 8,
            "the ninth one": 9,
            "the tenth one": 10,
            "the eleventh one": 11,
            "the twelfth one": 12,
            "first one": 1,
            "second one": 2,
            "third one": 3,
            "fourth one": 4,
            "fifth one": 5,
            "sixth one": 6,
            "seventh one": 7,
            "eighth one": 8,
            "ninth one": 9,
            "tenth one": 10,
            "eleventh one": 11,
            "twelfth one": 12,
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
            "tenth": 10,
            "eleventh": 11,
            "twelfth": 12,
            "the first": 1,
            "the second": 2,
            "the third": 3,
            "the fourth": 4,
            "the fifth": 5,
            "the sixth": 6,
            "the seventh": 7,
            "the eighth": 8,
            "the ninth": 9,
            "the tenth": 10,
            "the eleventh": 11,
            "the twelfth": 12,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }

        movies = []
        # Simple Disambiguate (part 2)
        most_recent_year = -10000000
        oldest_year = 10000000
        oldest_film = candidates[0]
        newest_film = candidates[0]

        for candidate in candidates:
            movie_title = self.titles[candidate][0]
            title, year = self.process_title_year(movie_title)

            if year and int(year[1: len(year) - 1]) < oldest_year:
                oldest_year = int(year[1: len(year) - 1])
                oldest_film = candidate

            if year and int(year[1: len(year) - 1]) > most_recent_year:
                most_recent_year = int(year[1: len(year) - 1])
                newest_film = candidate

            if (clarification in title) or ("(" + clarification + ")" == year):
                movies.append(candidate)

            elif clarification in ordering_words:
                movies.append(candidates[ordering_words[clarification] - 1])

            elif self.replace_filler_words(clarification) in title.lower():
                movies.append(candidate)

            elif clarification.isnumeric() and int(clarification) <= len(candidates):
                sequence_number = int(clarification) - 1
                movies.append(candidates[sequence_number])

        if clarification in want_newest_film_words:
            movies.append(newest_film)

        if clarification in want_oldest_film_words:
            movies.append(oldest_film)

        return movies

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

       To binarize a matrix, replace all entries above the threshold with 1.
       and replace all entries at or below the threshold with a -1.

       Entries whose values are 0 represent null values and should remain at 0.

       Note that this method is intentionally made static, as you shouldn't use
       any attributes of Chatbot like self.ratings in this method.

       :param ratings: a (num_movies x num_users) matrix of user ratings, from
        0.5 to 5.0
       :param threshold: Numerical rating above which ratings are considered
       positive

       :returns: a binarized version of the movie-rating matrix
       """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)
        num_rows = ratings.shape[0]
        num_cols = ratings.shape[1]
        for row in range(num_rows):
            for col in range(num_cols):
                value = ratings[row][col]
                if value == 0:
                    continue
                if value > threshold:
                    binarized_ratings[row][col] = 1
                else:  # 2.5 and below
                    binarized_ratings[row][col] = -1

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

       You may assume that the two arguments have the same shape.

       :param u: one vector, as a 1D numpy array
       :param v: another vector, as a 1D numpy array

       :returns: the cosine similarity between the two vectors
       """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
        filtering.

       You should return a collection of `k` indices of movies recommendations.

       As a precondition, user_ratings and ratings_matrix are both binarized.

       Remember to exclude movies the user has already rated!

       Please do not use self.ratings directly in this method.

       :param user_ratings: a binarized 1D numpy array of the user's movie
           ratings
       :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
         `ratings_matrix[i, j]` is the rating for movie i by user j
       :param k: the number of recommendations to generate
       :param creative: whether the chatbot is in creative mode

       :returns: a list of k movie indices corresponding to movies in
       ratings_matrix, in descending order of recommendation.
       """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        predictions = []
        rated_movies = np.where(user_ratings != 0)[0]
        unrated_movies = np.where(user_ratings == 0)[0]

        for cur_movie in range(len(ratings_matrix)):
            # calculate cosine sim weight for every other user
            # if cur_movie not in rated_movies:  # it's unrated
            if cur_movie in unrated_movies:
                val = 0
                if ratings_matrix[
                    cur_movie
                ].any():  # not just an empty row, at least 1 val
                    for rated in rated_movies:
                        addThis = user_ratings[rated] * self.similarity(
                            ratings_matrix[cur_movie], ratings_matrix[rated]
                        )
                        val = val + addThis
                    predictions.append((cur_movie, val))
                else:
                    predictions.append((cur_movie, 0))

        predictions.sort(key=lambda x: x[1], reverse=True)  # truncate from k

        for i in range(k):
            recommendations.append(predictions[i][0])  # predictions is (movie, val)

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
       Return debug information as a string for the line string from the REPL

       NOTE: Pass the debug information that you may think is important for
       your evaluators.
       """
        debug_info = "debug info"
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

       Consider adding to this description any information about what your
       chatbot can do and how the user can interact with it.
       """
        return """
       I'm a super hip bot that'll recommend you some fire movies.
       Tell me about movies you like or dislike with the movie names in quotes to get started!
       """


if __name__ == "__main__":
    print("To run your chatbot in an interactive loop from the command line, " "run:")
    print("    python3 repl.py")

