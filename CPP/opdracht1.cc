#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype>
#include <numeric>
#include <sstream>
#include <filesystem>

int main() {

     std::ifstream file("opdrachten/cpp1/dracula.txt");
     if (!file) {
         std::cerr << "Kon het bestand niet openen: dracula.txt" << std::endl;
         return 1;
     }
     
     std::cout << "Bestand succesvol geopend!" << std::endl;
    
    std::vector<char> characters;
    char ch;
    while (file.get(ch)) {
        characters.push_back(ch);
    }
    
    std::cout << "Aantal karakters: " << characters.size() << std::endl;
    
    size_t lineCount = std::count(characters.begin(), characters.end(), '\n') + 1;
    std::cout << "Aantal regels: " << lineCount << std::endl;
    
    
    size_t alphabeticCount = std::count_if(characters.begin(), characters.end(), 
                                           [](char c) { return std::isalpha(c); });
    std::cout << "Aantal alfabetische karakters: " << alphabeticCount << std::endl;
    
    
    bool onlyLettersAndPunctuation = std::all_of(characters.begin(), characters.end(),
                                                [](char c) { 
                                                    return std::isalpha(c) || 
                                                           std::ispunct(c) || 
                                                           std::isspace(c); 
                                                });
    std::cout << "Bevat alleen letters en leestekens: " 
              << (onlyLettersAndPunctuation ? "Ja" : "Nee") << std::endl;
    

    std::transform(characters.begin(), characters.end(), characters.begin(),
                  [](char c) { return std::tolower(c); });
    
    std::map<char, int> letterFrequency;
    for (char c = 'a'; c <= 'z'; ++c) {
        letterFrequency[c] = std::count(characters.begin(), characters.end(), c);
    }
    
    std::cout << "\nLetterfrequentie (gesorteerd op lettervolgorde):" << std::endl;
    for (const auto& pair : letterFrequency) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    std::vector<std::pair<char, int>> letterFreqVec(letterFrequency.begin(), letterFrequency.end());
    
    std::sort(letterFreqVec.begin(), letterFreqVec.end(),
             [](const std::pair<char, int>& a, const std::pair<char, int>& b) {
                 return a.second > b.second;
             });
    
    std::cout << "\nLetterfrequentie (gesorteerd op frequentie):" << std::endl;
    for (const auto& pair : letterFreqVec) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    file.clear();
    file.seekg(0);
    
    std::map<std::string, int> wordFrequency;
    std::string word;
    
    while (file >> word) {
        // Verwijder leestekens aan begin en einde van woorden
        while (!word.empty() && std::ispunct(word.front())) {
            word.erase(0, 1);
        }
        while (!word.empty() && std::ispunct(word.back())) {
            word.pop_back();
        }
        
        // Zet om naar kleine letters
        std::transform(word.begin(), word.end(), word.begin(),
                      [](char c) { return std::tolower(c); });
        
        // Tel alleen niet-lege woorden
        if (!word.empty()) {
            wordFrequency[word]++;
        }
    }
    
    // Maak een vector van de map voor sortering op frequentie
    std::vector<std::pair<std::string, int>> wordFreqVec(wordFrequency.begin(), wordFrequency.end());
    
    // Sorteer op frequentie (van hoog naar laag)
    std::sort(wordFreqVec.begin(), wordFreqVec.end(),
             [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                 return a.second > b.second;
             });
    
    // Druk de 10 meest voorkomende woorden af
    std::cout << "\nTop 10 meest voorkomende woorden:" << std::endl;
    for (size_t i = 0; i < std::min(wordFreqVec.size(), size_t(10)); ++i) {
        std::cout << i+1 << ". \"" << wordFreqVec[i].first << "\": " 
                  << wordFreqVec[i].second << " keer" << std::endl;
    }
    
    return 0;
}