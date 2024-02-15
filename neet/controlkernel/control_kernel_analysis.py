# control_kernel_analysis.py
#
# Bryan Daniels
# 5.15.2020
#
# Defining control kernel analyses to be run on networks
#
# Combining code from run_control_kernels.py and run_sampled_control_kernels.py
#

from . import modularity as md
import numpy as np
import time
#from neet.synchronous import Landscape # for basin size analysis

def ck_analysis(net,verbose=False):

    # () find attractors and associated control kernels

    before = time.time()
    print("Finding attractors and control kernels...")
    a,outdict = md.attractors(net,
                              find_control_kernel=True,
                              verbose=verbose)
    after = time.time()
    ck_time_minutes = (after - before)/60.

    ck_list = outdict['control_kernels']

    control_kernel_sizes = []
    for ck in ck_list:
        if ck is None: control_kernel_sizes.append(None)
        else: control_kernel_sizes.append( len(ck) )

    # () basin entropy analysis
    # for now just track which differential basin entropies
    # show up with which differential control kernel sizes
    delta_control_kernel_sizes,delta_basin_entropies = [],[]
    for module_index in outdict['delta_control_nodes'].keys():
        ck_nodes = outdict['delta_control_nodes'][module_index]
        basin_ents = outdict['basin_entropies'][module_index]
        delta_control_kernel_sizes.extend([ len(ck) for ck in ck_nodes if ck is not None])
        delta_basin_entropies.extend([ b for b,ck in zip(basin_ents,ck_nodes) if ck is not None])
    delta_control_kernel_sizes = np.array(delta_control_kernel_sizes)
    delta_basin_entropies = np.array(delta_basin_entropies)

    simple_control_entropies = np.all(delta_basin_entropies == delta_control_kernel_sizes)

    # () set output data

    data = {
        "size":net.size,
        "control_kernel_sizes":control_kernel_sizes,
        "control_kernels":ck_list,
        "attractors":a,
        "has_limit_cycles":np.any([len(att)>1 for att in a]),
        "delta_control_nodes":outdict['delta_control_nodes'],
        "basin_entropies":outdict['basin_entropies'],
        "delta_control_kernel_sizes":delta_control_kernel_sizes,
        "delta_basin_entropies":delta_basin_entropies,
        "simple_control_entropies":simple_control_entropies,
        "control_kernel_time_minutes":ck_time_minutes,
        "modules":outdict['modules'],
    }
    
    return data

def dn_analysis(net,attractors=None,possible_nodes_list=None,verbose=True,
    require_inputs=False,test_known_small_dns=False):
    """
    test_known_small_dns (False)            : If True, test dist. node sets already
                                              calculated for other attractors first,
                                              but only if they have size < log2(#atts).
                                              This is an attempt to quickly bound the
                                              mean below log2(#atts).
    """
    
    if require_inputs:
        required_nodes = tuple(md.input_nodes(net))
    else:
        required_nodes = tuple([])
    
    # () find attractors
    data = {"size":net.size,
            "require_inputs":require_inputs,
    }
    if attractors is None:
        before = time.time()
        print("Finding attractors...")
        a = md.attractors(net)
        after = time.time()
        att_time_minutes = (after - before)/60.
        data["attractors_time_minutes"] = att_time_minutes
    else:
        a = attractors
    decode = net.decode
    a_decoded = [ [ decode(state) for state in att ] for att in a ]
    data.update({"attractors":a,
                 "has_limit_cycles":np.any([len(att)>1 for att in a]),
    })

    # () find distinguishing nodes
    before = time.time()
    if (possible_nodes_list is None) and not test_known_small_dns:
        print("dn_analysis: Finding minimal distinguishing nodes...")
    else:
        print("dn_analysis: Finding bound on minimal distinguishing nodes...")
    
    dn_min_list,dn_min_sizes = distinguishing_nodes_from_atts_decoded(a_decoded,
                                required_nodes=required_nodes,
                                test_known_small_dns=test_known_small_dns,
                                possible_nodes_list=possible_nodes_list,
                                verbose=verbose)
    
    after = time.time()
    dn_time_minutes = (after - before)/60.

    if (possible_nodes_list is not None) or test_known_small_dns:
        data.update({
            "required_nodes":required_nodes,
            "test_known_small_dns":test_known_small_dns,
            "distinguishing_nodes_bound_possible_nodes_list":possible_nodes_list,
            "distinguishing_nodes_bound_sizes":dn_min_sizes,
            "distinguishing_nodes_bound":dn_min_list,
            "distinguishing_nodes_bound_time_minutes":dn_time_minutes,
            })
    elif require_inputs:
        data.update({
            "distinguishing_nodes_with_inputs_sizes":dn_min_sizes,
            "distinguishing_nodes_with_inputs":dn_min_list,
            "distinguishing_nodes_time_minutes":dn_time_minutes,
            })
    else:
        data.update({
            "distinguishing_nodes_sizes":dn_min_sizes,
            "distinguishing_nodes":dn_min_list,
            "distinguishing_nodes_time_minutes":dn_time_minutes,
            })
        
    return data


def distinguishing_nodes_from_atts_decoded(a_decoded,
    required_nodes=(),test_known_small_dns=False,possible_nodes_list=None,
    verbose=True,small_dn_min_list=[]):
    """
    """
    dn_list = []
    attractors_mean = np.array( [ np.mean(att,axis=0) for att in a_decoded ] )
    dn_min_sizes,dn_min_list = [],[]
    for att_index in range(len(a_decoded)):
        if possible_nodes_list is None:
            possible_nodes = None
        else:
            possible_nodes = possible_nodes_list[att_index]
            
        if test_known_small_dns:
            sets_to_test_first = small_dn_min_list
            print("distinguishing_nodes_from_atts_decoded: DEBUG: len(small_dn_min_list) = {}".format(len(small_dn_min_list)))
            #if len(small_dn_min_list) > 0:
            #    print("distinguishing_nodes_from_atts_decoded: DEBUG: small_dn_min_list[0] = {}".format(small_dn_min_list[0]))
            #    print("distinguishing_nodes_from_atts_decoded: DEBUG: small_dn_min_list[-1] = {}".format(small_dn_min_list[-1]))
        else:
            sets_to_test_first = []
            
        dn_gen = md.distinguishing_nodes_from_attractor(a_decoded,att_index,
            attractors_mean=attractors_mean,
            required_nodes=required_nodes,
            possible_nodes=possible_nodes,
            sets_to_test_first=sets_to_test_first)
            
        try:
            dn_min = dn_gen.__next__()
            dn_min_sizes.append( len(dn_min) )
            dn_min_list.append(dn_min)
            if (len(dn_min) < np.log2(len(a_decoded))) and (dn_min not in small_dn_min_list):
                small_dn_min_list.append(dn_min)
            if verbose:
                print("distinguishing_nodes_from_atts_decoded: Successful distinguishing node set of size {} for attractor index {}".format(len(dn_min),att_index))
                print("distinguishing_nodes_from_atts_decoded: Distinguishing node set: {}".format(dn_min))
        except StopIteration:
            dn_min_sizes.append(None)
            dn_min_list.append(None)
            if verbose:
                print("distinguishing_nodes_from_atts_decoded: No distinguishing node set for attractor index {}".format(att_index))
    
    return dn_min_list,dn_min_sizes


def sampled_ck_analysis(net,seed=130,samples=10000,phenotype='all',
    phenotype_only=False,require_inputs=True,iterative=False):

    # () find attractors and associated control kernels

    if iterative:
        print("Finding attractors and control kernels using sampling (iterative)...")
    else:
        print("Finding attractors and control kernels using sampling...")

    if phenotype == 'internal':
        phenotype_nodes = [ i for i in range(net.size) if i not in md.input_nodes(net) ]
    elif phenotype == 'all':
        phenotype_nodes = [ i for i in range(net.size) ]
    else:
        raise Exception
    a = md.sampled_attractors(net,samples,phenotype_list=phenotype_nodes,seed=seed)

    before = time.time()
    if iterative:
        ck_list,rounds_list = md.sampled_control_kernel(net,samples,seed=seed,
                                            verbose=True,
                                            phenotype=phenotype,
                                            phenotype_only=phenotype_only,
                                            require_inputs=require_inputs,
                                            iterative=True)
    else:
        ck_list = md.sampled_control_kernel(net,samples,seed=seed,verbose=True,
                                            phenotype=phenotype,
                                            phenotype_only=phenotype_only,
                                            require_inputs=require_inputs,
                                            iterative=False)
    after = time.time()
    time_minutes = (after - before)/60.

    control_kernel_sizes = []
    for ck in ck_list:
        if ck is None: control_kernel_sizes.append(None)
        else: control_kernel_sizes.append( len(ck) )

    # () set output data

    data = {
        "phenotype":phenotype,
        "phenotype_only":phenotype_only,
        "require_inputs":require_inputs,
        "sampled_seed":seed,
        "num_samples":samples,
        "size":net.size,
        "sampled_attractors":a,
        "sampled_has_limit_cycles":np.any([len(att)>1 for att in a]),
    }
    
    if iterative:
        data.update({
            "sampled_control_kernel_iterative_sizes":control_kernel_sizes,
            "sampled_control_kernels_iterative":ck_list,
            "sampled_iterative_rounds_list":rounds_list,
            "sampled_iterative_control_kernel_time_minutes":time_minutes,
            })
    else:
        data.update({
            "sampled_control_kernel_sizes":control_kernel_sizes,
            "sampled_control_kernels":ck_list,
            "sampled_exact_control_kernel_time_minutes":time_minutes,
            })
        
    return data
    
def sampled_dn_analysis(net,seed=130,samples=10000,phenotype='all',
    require_inputs=False,verbose=False,attractors=None):

    before = time.time()
    
    # () find distinguishing nodes

    print("Finding minimal distinguishing node sets using sampling...")
    
    if require_inputs:
        dn_list = md.sampled_distinguishing_nodes(net,samples,seed=seed,
                                    verbose=True,
                                    required_nodes=tuple(md.input_nodes(net)),
                                    phenotype=phenotype,
                                    attractors=attractors)
    else:
        dn_list = md.sampled_distinguishing_nodes(net,samples,seed=seed,
                                    verbose=True,
                                    phenotype=phenotype,
                                    attractors=attractors)
    
    #assert(len(dn_list)==len(dn_with_inputs_list))

    dn_min_sizes,dn_min_list = [],[]
    for i,dn in enumerate(dn_list):
        try:
            dn_min = dn.__next__()
            dn_min_sizes.append( len(dn_min) )
            dn_min_list.append(dn_min)
            if verbose:
                print("sampled_dn_analysis: Found distinguishing nodes of size {} for attractor with index {}".format(len(dn_min),i))
        except StopIteration:
            dn_min_sizes.append(None)
            dn_min_list.append(None)
        
    after = time.time()
    dn_time_minutes = (after - before)/60.

    # () set output data

    data = {
        "phenotype":phenotype,
        "sampled_seed":seed,
        "require_inputs":require_inputs,
        "num_samples":samples,
        "size":net.size,
        "sampled_distinguishing_nodes_time_minutes":dn_time_minutes,
        }
        
    if require_inputs:
        data.update({
            "sampled_distinguishing_nodes_with_inputs_sizes":dn_min_sizes,
            "sampled_distinguishing_nodes_with_inputs":dn_min_list,
                    })
    else:
        data.update({
            "sampled_distinguishing_nodes_sizes":dn_min_sizes,
            "sampled_distinguishing_nodes":dn_min_list,
                    })
        
    return data


def encode_input(state,inputNodes):
    if len(inputNodes) == 0:
        return 0
    else:
        return int(''.join(['0b']+[str(state[i]) for i in inputNodes]),2)


def split_data_on_input(data,net):
    """
    Given control kernel data from a network, split into 2^(# input nodes) sets of data
    corresponding to every setting of input nodes.
    """
    dataNewList = []
    inputNodes = md.input_nodes(net)
    
    # things are named differently for sampled data
    if 'attractors' in data:
        prefix = ''
    elif 'sampled_attractors' in data:
        prefix = 'sampled_'
    else:
        raise Exception('No attractor data found')
    
    decode = net.decode
    decoded_attractors = [ [ decode(state) for state in att ] for att in data[prefix+'attractors'] ]
    
    # splitDict maps encoded input identifiers to attractors with that input
    splitDict = dict([ (i,[]) for i in range(2**len(inputNodes)) ])
    for att_index,att in enumerate(decoded_attractors):
        encodedInput = encode_input(att[0],inputNodes) # use first state in attractor
        splitDict[encodedInput].append(att_index)
        
    for encodedInput,indices in splitDict.items():
        #assert(len(indices) > 0)
        if len(indices) == 0:
            print("split_data_on_input WARNING: No match for input value {} in network {}".format(encodedInput,data['name']))
        dataNew = {'name':data['name'],
                   'size':data['size'],
                   'encoded_input':encodedInput,
                   prefix+'attractors':[data[prefix+'attractors'][i] for i in indices]
                  }
        if 'threshold parameter' in data:
            dataNew['threshold parameter'] = data['threshold parameter']
        
        # define new control kernels that don't include input nodes
        if prefix+'control_kernels' in data:
            split_control_kernels,split_control_kernel_sizes = [],[]
            for ind in indices:
                ck = data[prefix+'control_kernels'][ind]
                if ck is not None:
                    ckSplit = set([ node for node in ck if node not in inputNodes ])
                    split_control_kernels.append(ckSplit)
                    split_control_kernel_sizes.append(len(ckSplit))
                else:
                    split_control_kernels.append(None)
                    split_control_kernel_sizes.append(None)
            dataNew[prefix+'control_kernels'] = split_control_kernels
            dataNew[prefix+'control_kernel_sizes'] = split_control_kernel_sizes
            
        # define new distinguishing node sets that don't include input nodes
        for dnName in [prefix+'distinguishing_nodes_with_inputs',
                       prefix+'distinguishing_nodes',
                       prefix+'distinguishing_nodes_bound',]:
            if dnName in data:
                split_dns,split_dn_sizes = [],[]
                for ind in indices:
                    dn = data[dnName][ind]
                    if dn is not None:
                        dnSplit = set([ node for node in dn if node not in inputNodes ])
                        split_dns.append(dnSplit)
                        split_dn_sizes.append(len(dnSplit))
                    else:
                        split_dns.append(None)
                        split_dn_sizes.append(None)
                # remove any reference to inputs in the name
                dnNameNew = dnName.replace('_with_inputs','')
                dataNew[dnNameNew] = split_dns
                dataNew[dnNameNew+'_sizes'] = split_dn_sizes
         
        dataNewList.append(dataNew)
        
    return dataNewList


# 2020.9.29
def dn_analysis_split_on_input(data,net,
    test_known_small_dns=False,restrict_to_cks=False):
    """
    Given existing bounds on distinguishing node sets, improve bounds on
    distinguishing node sets limited to specific input states to attempt to
    get the bound below log2(ri), where ri is the number of attractors given
    the input state.
    """
    decode = net.decode
    
    dataSplitList = split_data_on_input(data,net)
    
    # loop over input states
    for dataSplit in dataSplitList:
        
        if np.mean(dataSplit["distinguishing_nodes_bound_sizes"]) > \
           np.log2(len(dataSplit["attractors"])):
            
            a_decoded = [ [ decode(state) for state in att ] \
                          for att in dataSplit["attractors"] ]
            
            # use existing small distinguishing node sets when they are below log2(ri)
            small_dn_min_list = \
                [ dn for dn in dataSplit["distinguishing_nodes_bound"] \
                  if len(dn) < np.log2(len(a_decoded)) ]
            
            if restrict_to_cks:
                possible_nodes_list = dataSplit["control_kernels"]
            else:
                possible_nodes_list = None
            
            dn_min_list,dn_min_sizes = distinguishing_nodes_from_atts_decoded(a_decoded,
                test_known_small_dns=test_known_small_dns,
                possible_nodes_list=possible_nodes_list,
                small_dn_min_list=small_dn_min_list)
                
            if (possible_nodes_list is not None) or test_known_small_dns:
                dataSplit["distinguishing_nodes_bound"] = dn_min_list
                dataSplit["distinguishing_nodes_bound_sizes"] = dn_min_sizes
            else:
                dataSplit["distinguishing_nodes"] = dn_min_list
                dataSplit["distinguishing_nodes_sizes"] = dn_min_sizes
            
    return dataSplitList


#def basin_analysis(net):
#
#    # () find attractors and associated basin sizes
#
#    before = time.time()
#    print("Finding attractors and basin sizes...")
#    ls = Landscape(net)
#    atts = ls.attractors
#    basin_sizes = ls.basin_sizes
#    basin_entropy = ls.basin_entropy()
#    after = time.time()
#    basin_time_minutes = (after - before)/60.
#
#    # () set output data
#
#    data = {
#        "size":net.size,
#        "attractors":atts,
#        "basin_sizes":basin_sizes,
#        "basin_entropy":basin_entropy,
#        "basin_time_minutes":basin_time_minutes,
#    }
#
#    return data
