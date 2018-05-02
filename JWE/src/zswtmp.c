#include <stdio.h>
int main(){

      printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
      printf("Options:\n");
      printf("Parameters for training:\n");
      printf("\t-train <file>\n");
      printf("\t\tUse text data from <file> to train the model\n");
      printf("\t-output-word <file>\n");
      printf("\t\tUse <file> to save the resulting word vectors\n");
      printf("\t-output-char <file>\n");
      printf("\t\tUse <file> to save the resulting character vectors\n");
      printf("\t-output-comp <file>\n");
      printf("\t\tUse <file> to save the resulting component vectors\n");
      printf("\t-size <int>\n");
      printf("\t\tSet size of word vectors; default is 200\n");
      printf("\t-window <int>\n");
      printf("\t\tSet max skip length between words; default is 5\n");
      printf("\t-sample <float>\n");
      printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
      printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
      printf("\t-negative <int>\n");
      printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
      printf("\t-iter <int>\n");
      printf("\t\tRun more training iterations (default 5)\n");
      printf("\t-threads <int>\n");
      printf("\t\tUse <int> threads (default 1)\n");
      printf("\t-min-count <int>\n");
      printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
      printf("\t-alpha <float>\n");
      printf("\t\tSet the starting learning rate; default is 0.025\n");
      printf("\t-debug <int>\n");
      printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
      printf("\t-binary <int>\n");
      printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
      printf("\t-comp <file>\n");
      printf("\t\tUse component list from <file>\n");
      printf("\t-char2comp <file>\n");
      printf("\t\tObtain the mapping between characters and their components from <file>\n");
      printf("\t-join-type <int>\n");
      printf("\t\t The type of methods combining subwords (default = 1: sum loss, 2 : sum context)\n");
      printf("\t-pos-type <int>\n");
      printf("\t\t The type of subcomponents' positon (default = 1: use the components of surrounding words, 2: use the components of the target word, 3: use both)\n");
      printf("\t-average-sum <int>\n");
      printf("\t\t The type of compositional method for context (average_sumdefault = 1 use average operation, 0 use sum operation)\n");
      printf("\nExamples:\n");
      printf("./jwe -train wiki_process.txt -output-word word_vec -output-char char_vec -output-comp comp_vec -size 200 -window 5 -sample 1e-4 -negative 10 -iter 100 -threads 8 -min-count 5 -alpha 0.025 -binary 0 -comp ../subcharacter/comp.txt -char2comp ../subcharacter/char2comp.txt -join-type 1 -pos-type 1 -average-sum 1\n\n");
}
