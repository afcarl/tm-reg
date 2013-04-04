#include <iterator>
#include <vector>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <boost/unordered_map.hpp>

#include "lemur/IndexTypes.hpp"

typedef boost::unordered_map<lemur::api::TERMID_T, std::pair<float, float> >::iterator FastTargetIterator;
typedef boost::unordered_map<lemur::api::TERMID_T, boost::unordered_map<lemur::api::TERMID_T, std::pair<float, float> > >::iterator FastSourceIterator;
typedef boost::unordered_map<lemur::api::TERMID_T, float>::iterator FastTotalIterator;
typedef std::vector< std::pair< std::vector<lemur::api::TERMID_T>, std::vector<lemur::api::TERMID_T> > > AlignedDataset;
typedef boost::unordered_map<lemur::api::TERMID_T, boost::unordered_map<lemur::api::TERMID_T, std::pair<float, float> > > FastTrainingTable;
typedef boost::unordered_map<lemur::api::TERMID_T, float> FastTotal;

// Translation Model final structure  
typedef std::map<lemur::api::TERMID_T, std::vector< std::pair<lemur::api::TERMID_T, float> > > TranslationModel;

// Train EM IBM Model 1
void train_em_model1(AlignedDataset const&, int, TranslationModel &);
