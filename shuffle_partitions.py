import pandas as pd



df = pd.read_csv('training_set.csv')
shuffle_state = 1
# shuffle_state_history: 42, 

# !NOTICE: when you want to run a new batch of the dataframe, please change the shuffle_state and add the new state in the shuffle_state_history comment 
# Shuffle all the data points - keep 100% of the data (frac=1)
df = df.sample(frac=1, random_state=shuffle_state).reset_index(drop=True)


num_partitions = 5
partition_size = len(df) // num_partitions
shuffled_partitions = []

for i in range(num_partitions):
    start = i * partition_size
    end = (i + 1) * partition_size if i != num_partitions - 1 else len(df)
    
    # partition and the remain data parts
    partition_df = df.iloc[start:end]
    before_partition_df = df.iloc[:start]
    after_partition_df = df.iloc[end:]
    
    # shuffle within each partition independently - only take all 90% of the data for each partition 
    partition_df = partition_df.sample(frac=0.9, random_state= shuffle_state + i).reset_index(drop=True)
    
    # join the shuffled partition with the remaining data
    rejoin_df = pd.concat([partition_df, before_partition_df, after_partition_df], ignore_index=True)

    # save each rejoined dataframe to a separate CSV file
    letter = chr(65 + i)  

    # we dont need more than 26 versions of training set, this is just in case
    if ord(letter) > 90: 
        print("More than 26 versions of training set are created, duplicate name error")

    # naming versions as letters to prevent erros in GC buckets
    rejoin_df.to_csv(f"rejoin_training_set_S{shuffle_state}V{letter}.csv", index=False)

