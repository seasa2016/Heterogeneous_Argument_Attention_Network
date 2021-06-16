for t in 'tree' 'tree_bert'
do
    mkdir ${1}/${t}/
    mkdir ${1}/${t}/${2}/
    mkdir ${1}/${t}/${2}/train

    cp ${1}/train/${t}/train/* ${1}/${t}/${2}/
    for d in 'data' 'graph_pair'
    do
        cp ${1}/heldout/${t}/train/${d} ${1}/${t}/${2}/${d}_test
    done
done

for t in 'neg' 'pos'
do
    cp ${1}/train/tree/cmv_elmo_${t}.hdf5 ${1}/tree/cmv_elmo_all_${t}.hdf5
    cp ${1}/heldout/tree/cmv_elmo_${t}.hdf5 ${1}/tree/cmv_elmo_test_${t}.hdf5
done
