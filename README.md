# AdaRank
## GUI used for experiments

This app uses [Streamlit](https://docs.streamlit.io/) which is an open-source python library for creating custom web-apps for data science.

### Rule Mining Process
For creating rules please refer to ER-Miner implemented in [SPMF](https://www.philippe-fournier-viger.com/spmf/ERMiner.php) by Prof. Fournier-Viger 
and [SCORER-Gap](https://github.com/nimbus262/scorer-gap).

### Simulation Process
Each simulation gets put together based on the configurations made on the page:
1. One has to upload a rule set which is the output of the aforementioned rule mining algorithm2
2. Text files containing the item sequences are required
   1. The first uploaded files contain the training set of item sequences
   2. The second one contains the test set
3. The first dropdown menu on the left hand side lets you choose the selection method(s). If CGap is among those methods another two files are required to be uploaded; the first one is the file containing timestamp sequences matching each sequence and each item in the item sequence files.
4. The next input field lets you select the number of sequence which should be simulated.
5. And in the last input field the number of runs can be configured.
6. On the right hand side the first slider configures the k value of top-k rules. Here, the number of rules for recommendation is meant.
7. If the checkbox "Only recommend last event?" is checked, then experiments with next-item-recommendation are conducted. 
8. If not, the successive-item-recommendation is performed. In this case, the prefix length of the sequence has to be chosen as a recommendation basis.

After running the simulation process you get two types of files for each parameter configuration:
1. A CSV-file which contains the results in the form of precision and diversity values and the corresponding parameters for this experiment.
2. A textfile containing the simulated sequences.

If you need help regarding the input file format please have a look at the help page once you started the GUI.

