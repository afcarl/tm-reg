/*
 * train_em_model1(): EM Process for model 1
 */

#include "em.hpp"

using namespace lemur::api;
using namespace std;

// Fill in translationModel
void train_em_model1(AlignedDataset const& dataset,
					 int maxiter,
                     TranslationModel & translationModel) {
  
  FastTrainingTable trainingTable;
  FastTotal totalSource;

  // Count co-occurrences c(target_term, source_term)
  for (int ex = 0; ex < dataset.size(); ex++) {
    std::vector<TERMID_T> targetVector = dataset[ex].first;
    std::vector<TERMID_T> sourceVector = dataset[ex].second;

    for (int q = 0; q < sourceVector.size(); q++) {
      // emplace table
      boost::unordered_map<TERMID_T, pair<float, float> > targets;
      std::pair<FastSourceIterator, bool> ret = trainingTable.emplace(sourceVector[q], targets);
      // emplace source count
      totalSource.emplace(sourceVector[q], 0);

      // count co-occurrences
      boost::unordered_map<TERMID_T, pair<float, float> > & refTargets = ret.first->second;
      for (int t = 0; t < targetVector.size(); t++) {
        FastTargetIterator twit = refTargets.find(targetVector[t]);
        if (twit != refTargets.end()) {
          twit->second.first++;
        } else {
          // counts to 0, prob to 1
          refTargets.emplace(targetVector[t], make_pair(1, 0));
        }
      }
    }
  }

  // Normalize
  FastSourceIterator qwit = trainingTable.begin();
  for (qwit; qwit != trainingTable.end(); qwit++) {
    float sum = 0.;
    FastTargetIterator twit = qwit->second.begin();
    for (twit; twit != qwit->second.end(); twit++)
      sum += twit->second.first;

    twit = qwit->second.begin();
    for (twit; twit != qwit->second.end(); twit++)
      twit->second.first /= sum;
  }

  int niter = 0;
  // We are ready to train
  while (niter++ < maxiter) {
    cerr << "it # " << niter << endl;

    // INITIALIZE STRUCTURES
    FastSourceIterator sourceWord = trainingTable.begin();
    for (sourceWord; sourceWord != trainingTable.end(); sourceWord++) {
      totalSource.find(sourceWord->first)->second = 0.;

      // REINITIALIZE COUNTS
      FastTargetIterator targetWord = sourceWord->second.begin();
	  for (targetWord; targetWord != sourceWord->second.end(); targetWord++) {
        targetWord->second.second = 0.;
      }
    }
	
    for (int ex = 0; ex < dataset.size(); ex++) {
      // Log
	  cerr << "Example # " << ex << "\r";

      // Compute normalization
      const std::vector<TERMID_T>& target = dataset[ex].first;
      const std::vector<TERMID_T>& source = dataset[ex].second;

      FastTotal totalTarget;
	  
	  // Hack to avoid map seeks
      std::vector<float> targetSum(target.size());
      
	  for (int q = 0; q < source.size(); q++) {
        FastSourceIterator sourceWord = trainingTable.find(source[q]);
		
        for (int t = 0; t < target.size(); t++) {
          // Add probability
          targetSum[t] += sourceWord->second.find(target[t])->second.first;
        }
      }

      for (int t = 0; t < target.size(); t++) {
        totalTarget.insert(make_pair(target[t], targetSum[t]));
      }
	  
      for (int q = 0; q < source.size(); q++) {
        FastSourceIterator sourceWord = trainingTable.find(source[q]);
        FastTotalIterator totalSourceIt = totalSource.find(source[q]);
		
        for (int t = 0; t < target.size(); t++) {
          FastTotalIterator totalTargetIt = totalTarget.find(target[t]);
          FastTargetIterator targetWord = sourceWord->second.find(target[t]);

          float res = targetWord->second.first / totalTargetIt->second;
          // count(target, source) += res
          targetWord->second.second += res;
          
		  // total(source) += res
          totalSourceIt->second += res;
        }
      }
    }

    // M-step and renormalize
    sourceWord = trainingTable.begin();
    for (sourceWord; sourceWord != trainingTable.end(); sourceWord++) {
      FastTargetIterator targetWord = sourceWord->second.begin();
      FastTotalIterator totalSourceIt = totalSource.find(sourceWord->first);
	  
      for (targetWord; targetWord != sourceWord->second.end(); targetWord++) {
        targetWord->second.first = targetWord->second.second/totalSourceIt->second;
      }
    }
  }

  // Copy the model into a lighter form
  FastSourceIterator sourceWord = trainingTable.begin();
  for (sourceWord; sourceWord != trainingTable.end(); sourceWord++) {
	FastTargetIterator targetWord = sourceWord->second.begin();
	
	std::vector<pair<TERMID_T, float> > targets;
	for (targetWord; targetWord != sourceWord->second.end(); targetWord++) {
	  targets.push_back(make_pair(targetWord->first, targetWord->second.first));
	}
	
	// Free space
	sourceWord->second.clear();
	// Fill in the TM
	translationModel.insert(make_pair(sourceWord->first, targets));
  }
  
  // Free all
  trainingTable.clear();
  // Done
}
//
