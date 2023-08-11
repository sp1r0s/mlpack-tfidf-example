// Define these to print extra informational output and warnings.
#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#include <mlpack/mlpack.hpp>
#include <filesystem>
#include <wchar.h>
#include <regex>

using namespace std;
using namespace mlpack;
using namespace mlpack::data;

namespace fs = std::filesystem;

using DictionaryType = StringEncodingDictionary<MLPACK_STRING_VIEW>;

class MlpackExample 
{	
	private:
		string class0Label;
		string class1Label;
		LogisticRegression<> model;
		vector<string> docs;
		arma::u64_rowvec labels;
		map<string, int> labelToClassIndexMap;
		TfIdfEncoding<SplitByAnyOf::TokenType> encoder = TfIdfEncoding<SplitByAnyOf::TokenType>(TfIdfEncodingPolicy::TfTypes::TERM_FREQUENCY, true);
		SplitByAnyOf tokenizer = SplitByAnyOf(" ");

		void cleanText(std::string& text) {
			transform(text.begin(), text.end(), text.begin(), ::tolower);
			std::regex whitespace(R"(\s+)");
			std::regex numbers(R"([0-9])");
			std::regex special(R"([><(\t)\(\)\-,\/])");
			text = std::regex_replace(text, numbers, "");
			text = std::regex_replace(text, special, " ");
			text = std::regex_replace(text, whitespace, " ");
		}

		void cleanTextForPrediction(std::string& text,
			DictionaryType const& dictionary,
			SplitByAnyOf const& tokenizer) {
			cleanText(text);

			MLPACK_STRING_VIEW strView(text);
			auto token = tokenizer(strView);
			string processedText = "";
			while (!tokenizer.IsTokenEmpty(token))
			{
				/* MLPACK encoder will expand the dictionary if unknown tokens are present in the prediction - text.
				 * To avoid that, simply remove the unknown ones.*/ 
				if (dictionary.HasToken(token)) {
					processedText.append(" ").append(token);
				}

				token = tokenizer(strView);
			}
			text = processedText;
		}

	public:
		MlpackExample(string label0, string label1)
		{
			class0Label = label0;
			class1Label = label1;
			labelToClassIndexMap.insert(pair<string, int>(class0Label, 0));
			labelToClassIndexMap.insert(pair<string, int>(class1Label, 1));
		}

		void loadData(string path) 
		{
			docs = vector<string>();
			vector<int> docTags = vector<int>();
			
			for (const auto& dataCategoryDir : fs::directory_iterator(path))
			{
				const string dataCategory = dataCategoryDir.path().string().substr(dataCategoryDir.path().string().find_last_of(filesystem::path::preferred_separator) + wcslen(&filesystem::path::preferred_separator), dataCategoryDir.path().string().length());
				for (const auto& textFile : fs::directory_iterator(dataCategoryDir.path()))
				{
					fstream file;
					file.open(textFile.path(), ios::in);
					if (file.is_open()) {
						string content;
						while (getline(file, content)) {
							//cout << content << "\n"; 
						}
						cleanText(content);
						docs.push_back(content);
						docTags.push_back(labelToClassIndexMap[dataCategory]);
						file.close();
					}
				}
			}
			
			labels = arma::conv_to<arma::u64_rowvec>::from(docTags);
		}

		void train() 
		{	
			arma::mat trainingData;
			encoder.Encode(docs, trainingData, tokenizer);
			const DictionaryType& dictionary = encoder.Dictionary();
			model = LogisticRegression<>(dictionary.Size());
			mlpack::ShuffleData(trainingData, labels, trainingData, labels);
			model.Train(trainingData, labels);
		}

		void predict(string text) 
		{
			const DictionaryType& dictionary = encoder.Dictionary();
			cleanTextForPrediction(text, dictionary, tokenizer);
			arma::mat inputSample;
			encoder.Encode({ text }, inputSample, tokenizer);
			arma::Row<size_t> predictions;
			arma::mat probabilities;
			model.Classify(inputSample, probabilities);
			std::cout << "\n\nProbabilities" <<
				"\n" + class0Label + ": " << probabilities.at(labelToClassIndexMap[class0Label]) <<
				"\n" + class1Label + ": " << probabilities.at(labelToClassIndexMap[class1Label]) << endl;
		}
};

int main() 
{
	MlpackExample example("spam", "non_spam");
	example.loadData("data/training");
	example.train();
	example.predict("congratulation you won money!");
}
