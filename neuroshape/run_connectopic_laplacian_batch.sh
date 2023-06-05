#!/bin/bash

#~ND~FORMAT~MARKDOWN~
#~ND~START~
#
# # run_connectopic_laplacian_batch.sh
#
# ## Copyright Notice
#
# Copyright (C) 2023 Systems Neuroscience Group Newcastle
#
# ## Author(s)
#
# * Nikitas C. Koussis, School of Psychological Sciences,
#   University of Newcastle
#
#
# ## License
#
# See the [LICENSE](https://github.com/breakspear/blob/main/LICENSE) file
#
# ## Description:
#
# Example script for running the connectopic Laplacian python script
# (connectopic_laplacian.py) over a directory of subjects
#
# ## Prerequisites
#
# ### Installed software
#
# * FSL
# * FreeSurfer
# * MRtrix3
#
# ### Environment variables
#
# Should be set in script file pointed to by EnvironmentScript variable.
# See setting of the EnvironmentScript variable in the main() function
# below.
#
# * FSLDIR - main FSL installation directory
# * FREESURFER_HOME - main FreeSurfer installation directory
# * MRTRIX - main MRtrix3 installation directory
# * PATH - must point to where MATLAB binary is located (Usually C://Program\ Files/
#   MATLAB_<version>/MATLAB.exe or /usr/local/matlab<version>/bin/matlab)
#
# <!-- References -->
# [neuroshape] : https://github.com/breakspear/neuroshape
#
#~ND~END~

# Function: get_batch_options
# Description
#
#   Retrieve the following command line parameter values if specified
#
#   --StudyFolder= - primary study folder containing subject ID subdirectories
#                    VERY IMPORTANT NOTE!: THIS MUST BE AN ABSOLUTE PATH I.E.,
#                    NOT QUOTED RELATIVE FROM THE CURRENT WORKING DIRECTORY, OTHERWISE
#                    THIS CODE WILL NOT WORK!
#   --Subjlist=    - quoted, space separated list of subject IDs on which
#                    to run the pipeline
#   --Structures=  - quoted, space separated list of ROI images on which to run
#                    to run the pipeline. IMPORTANT: all ROIs must be in MNI152 1mm or 2mm
#                    space, otherwise the registration step will fail. All ROIs must
#                    also be binary label images - i.e., have voxel value of 1 where ROI is
#                    and 0 elsewhere. They must also not have an extension when given 
#                    to the command, but have .nii.gz or .nii extension otherwise. 
#                    This script can also handle a single input 
#                    ROI via either quoted:
#                    --Structures="roi"
#                    or unquoted
#                    --structure=roi   
#
#   Set the values of the following global variables to reflect command
#   line specified parameters
#
#   command_line_specified_study_folder
#   command_line_specified_subj_list
#   command_line_specified_structures
#   command_line_specified_num_gradients
#
#   These values are intended to be used to override any values set
#   directly within this script file
get_batch_options() {
	local arguments=("$@")

	unset command_line_specified_study_folder
	unset command_line_specified_subj
	unset command_line_specified_structures
	unset command_line_specified_num_gradients
	unset command_line_specified_smoothing
	unset command_line_specified_filtering

	local index=0
	local numArgs=${#arguments[@]}
	local argument

	while [ ${index} -lt ${numArgs} ]; do
		argument=${arguments[index]}

		case ${argument} in
			--StudyFolder=*)
				command_line_specified_study_folder=${argument#*=}
				index=$(( index + 1 ))
				;;
            --fs_subjects_dir=*)
                command_line_specified_FreeSurfer_folder=${argument#*=}
                index=$(( index + 1 ))
                ;;
			--Subject=*)
				command_line_specified_subj=${argument#*=}
				index=$(( index + 1 ))
				;;
            --structure=*)
                command_line_specified_structures=${argument#*=}
                index=$(( index + 1 ))
                ;;
            -n=*|--n_gradients=*)
                command_line_specified_num_gradients=${argument#*=}
                index=$(( index + 1 ))
                ;;
            -s=*|--smooth=*)
                command_line_specified_smoothing=${argument#*=}
                index=$(( index + 1 ))
                ;;
            --filter)
                command_line_specified_filtering="TRUE"
                index=$(( index + 1 ))
                ;;			
			*)
				echo ""
				echo "ERROR: Unrecognized Option: ${argument}"
				echo ""
				exit 1
				;;
		esac
	done
}

main()
{
	get_batch_options "$@"

	# Set variable values that locate and specify data to process
	StudyFolder="/Volumes/Scratch/functional_integration_psychosis/preprocessed/HCP-EP/fmri/" # Location of Subject folders (named by subjectID)
	fs_subjects_dir="/Volumes/Scratch/functional_integration_psychosis/preprocessing/HCP-EP/FS"
	Subjlist="sub-1001 sub-1002 sub-1003 sub-1005 sub-1006 sub-1009 \
	sub-1010 sub-1012 sub-1015 sub-1017 sub-1018 sub-1019 sub-1020 \
	sub-1021 sub-1022 sub-1024 sub-1025 sub-1026 sub-1027 sub-1028 \
	sub-1029 sub-1030 sub-1031 sub-1032 sub-1033 sub-1034 sub-1035 \
	sub-1036 sub-1037 sub-1038 sub-1039 sub-1040 sub-1041 sub-1043 \
	sub-1044 sub-1045 sub-1047 sub-1048 sub-1050 sub-1051 sub-1052 \
	sub-1053 sub-1054 sub-1056 sub-1057 sub-1060 sub-1061 sub-1063 \
	sub-1064 sub-1065 sub-1066 sub-1067 sub-1068 sub-1070 sub-1071 \
	sub-1072 sub-1073 sub-1074 sub-1075 sub-1076 sub-1077 sub-1078 \
	sub-1079 sub-1080 sub-1081 sub-1082 sub-1083 sub-1084 sub-1085 \
	sub-1086 sub-1087 sub-1088 sub-1089 sub-1091 sub-1093 sub-1094 \
	sub-1095 sub-1098 sub-1099 sub-1104 sub-1105 sub-2004 sub-2005 \
	sub-2006 sub-2007 sub-2008 sub-2010 sub-2012 sub-2014 sub-2015 \
	sub-2016 sub-2019 sub-2020 sub-2022 sub-2023 sub-2029 sub-2031 \
	sub-2033 sub-2038 sub-2040 sub-2041 sub-2042 sub-2044 sub-2045 \
	sub-2049 sub-2052 sub-2062 sub-2065 sub-3002 sub-3009 sub-3011 \
	sub-3017 sub-3020 sub-3021 sub-3022 sub-3025 sub-3026 sub-3027 \
	sub-3028 sub-3029 sub-3031 sub-3032 sub-3034 sub-3035 sub-3039 \
	sub-4002 sub-4003 sub-4004 sub-4005 sub-4006 sub-4010 sub-4011 \
	sub-4012 sub-4014 sub-4015 sub-4018 sub-4022 sub-4023 sub-4024 \
	sub-4027 sub-4028 sub-4029 sub-4030 sub-4031 sub-4035 sub-4036 \
	sub-4037 sub-4038 sub-4040 sub-4047 sub-4048 sub-4049 sub-4050 \
	sub-4052 sub-4053 sub-4057 sub-4058 sub-4059 sub-4063 sub-4064 \
	sub-4065 sub-4069 sub-4071 sub-4072 sub-4075 sub-4088 sub-4091"                                # Space delimited list of subject IDs

    Structures="/Volumes/Scratch/functional_integration_psychosis/code/neuroshape/neuroshape/masks/striatum_2mm"
    num_gradients=30
    fwhm=0
    filter="FALSE"
    
	# Use any command line specified options to override any of the variable settings above
	if [ -n "${command_line_specified_study_folder}" ]; then
		StudyFolder="${command_line_specified_study_folder}"
	fi
	
	if [ -n "${command_line_specified_FreeSurfer_folder}" ]; then
		fs_subjects_dir="${command_line_specified_FreeSurfer_folder}"
	fi

	if [ -n "${command_line_specified_subj}" ]; then
		Subjlist="${command_line_specified_subj}"
	fi
	
	if [ -n "${command_line_specified_structure}" ]; then
		Structures="${command_line_specified_structure}"
	fi
	
	if [ -n "${command_line_specified_num_gradients}" ]; then
		num_gradients="${command_line_specified_num_gradients}"
	fi
	
	if [ -n "${command_line_specified_smoothing}" ]; then
		fwhm="${command_line_specified_smoothing}"
	fi
	
	if [ -n "${command_line_specified_filtering}" ]; then
		filter="TRUE"
	fi

	# Report major script control variables to user
	echo "StudyFolder: ${StudyFolder}"
	echo "FreeSurfer Folder: ${fs_subjects_dir}"
	echo "Subjlist: ${Subjlist}"
	echo "Structures: ${Structures}"
	echo "Number of gradients: ${num_gradients}"
	
	echo "Sourcing $(pwd) on the path"
	pipedir=$(pwd)
	
	SUBJECTS_DIR=${fs_subjects_dir}
	# mask folder
	MASK=$(pwd)/masks/GMmask.nii
	# Cycle through specified subjects
	for Subject in $Subjlist ; do
		echo $Subject

		cd ${StudyFolder}/${Subject}

		# Detect aseg.mgz images and build list of full paths
		numsegws=`ls ${fs_subjects_dir}/${Subject}/mri/aparc+aseg.mgz | wc -l`
		echo "Found ${numsegws} FreeSurfer Segmentation Images for subject ${Subject}"
		if [ $numsegws -gt 1 ]; then
    		echo ""
            echo "ERROR: Too many segmentation images: ${numsegws}"
            echo ""
            exit 1
        fi
        
        # Detect resting-state fmri images and build list of full paths
        numrestws=`ls ${StudyFolder}/${Subject} | grep 'bold' | wc -l`
        if [[ -z ${numrestws} ]]; then
            echo ""
            echo "ERROR: No fMRI Images found for subject ${subject}"
            echo ""
            exit 1
        fi
            
        echo "Found ${numrestws} fMRI Images for subject ${Subject}"
        fMRIInputImages=`ls *rest*bold.nii`
        InputImages=${fMRIInputImages}
        echo "fMRI input images: ${InputImages}"
        #i=1
        #while [ $i -le $numrestws ] ; do
        #    fMRIInputImages+=' *rest*bold.nii'
        #    i=$(($i+1))
        #done
        
        output_folder=${StudyFolder}/${Subject}/gradients
        mkdir -p gradients
        cd gradients
        
		segInputImage="${fs_subjects_dir}/${Subject}/mri/aparc+aseg.mgz"
		segConverted="aparc+aseg_nat.nii.gz"
		echo "Converting Input Image to Subject space : ${segInputImage} to ${segConverted}"
		tkregister2 --mov ${SUBJECTS_DIR}/${Subject}/mri/brain.mgz --targ ${SUBJECTS_DIR}/${Subject}/mri/rawavg.mgz --reg register.native.dat --noedit --regheader --fslregout FS2FSL.mat
		mri_vol2vol --mov ${SUBJECTS_DIR}/${Subject}/mri/brain.mgz --targ ${SUBJECTS_DIR}/${Subject}/mri/rawavg.mgz --regheader --o brainFSnat.nii
		
		mri_vol2vol --mov ${segInputImage} --targ ${SUBJECTS_DIR}/${Subject}/mri/rawavg.mgz --regheader --o ${segConverted} --nearest --keep-precision
	    labelconvert ${segConverted} $FREESURFER_HOME/FreeSurferColorLUT.txt $MRTRIX/share/mrtrix3/labelconvert/fs_default.txt parc.nii -nthreads 1 -force -quiet
	    mrconvert ${SUBJECTS_DIR}/${Subject}/mri/brain.mgz brainFS.nii -quiet -force -nthreads 1
	    
	    echo "Making GMmask.nii from segmentation image ${segConverted}"
        
		# Do connectopic Laplacian
		#cycle through structures
		for structure in ${Structures} ; do
    		rsync -aW ${structure}.nii* .
    		structure=$(basename $structure)
    		echo "Registering GMmask.nii to MNI152 space"
    		vox_size=`mrinfo ${structure}.nii* | grep Vox | cut -d " " -f12`
    		dims=`mrinfo ${structure}.nii* | grep Dim | cut -d " " -f11`
    		dims+=" `mrinfo ${structure}.nii* | grep Dim | cut -d " " -f13`"
    		dims+=" `mrinfo ${structure}.nii* | grep Dim | cut -d " " -f15`"
    		
    		if [[ ! -f FS2MNI.mat ]]; then
        		if [[ ${vox_size} == '2' ]]; then
            		standard=${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz
            		if [[ ${dims} == '91 109 91' ]]; then
                		flirt -ref ${FSLDIR}/data/standard/MNI152_T1_2mm_brain -in brainFS -dof 12 -cost normmi -omat FS2MNI.mat
                	else
                    	echo ""
        				echo "ERROR: ${structure} Must have the same dimensions as ${standard}, exiting."
        				echo ""
        				exit 1
        			fi	
            	elif [[ ${vox_size} == '1' ]]; then
                	standard=${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz
                	if [[ ${dims} == '182 218 182' ]]; then
                    	flirt -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain -in brainFS -dof 12 -cost normmi -omat FS2MNI.mat
                    else
                        echo ""
        				echo "ERROR: ${structure} Must have the same dimensions as ${standard}, exiting."
        				echo ""
        				exit 1
        			fi	
                else
                    echo ""
    				echo "ERROR: Voxel size must be 2mm or 1mm isotropic: ${structure}, exiting."
    				echo ""
    				exit 1
                fi
            fi
            
            mrconvert parc.nii parcstr.nii -strides +1,2,3 -force -quiet
            mv parcstr.nii parc.nii
            convert_xfm -omat FSL2FS.mat -inverse FS2FSL.mat
            flirt -ref brainFS.nii -in parc -applyxfm -init FSL2FS.mat -out parc_FS
            flirt -ref ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz -in parc_FS -applyxfm -init FS2MNI.mat -out parc_mni
            
            fslmaths parc_mni -bin GMmask
            fslmaths GMmask.nii.gz -ero GMmask_ero.nii.gz
            mv GMmask_ero.nii.gz GMmask.nii.gz
            gunzip -f GMmask.nii.gz
            mrconvert GMmask.nii GMmaskstr.nii -strides +1,2,3 -force -quiet
            mv GMmaskstr.nii GMmask.nii
            
        	#flirt -ref brainFS -in ${structure} -applyxfm -init MNI2FS.mat -out ${structure}_FS
    		#flirt -ref brainFSnat -in ${structure}_FS -applyxfm -init FS2FSL.mat -out ${structure}_nat
        	
        	for input_image in ${InputImages} ; do
            	rsync -aW ${StudyFolder}/${Subject}/${input_image} ${input_image}
        	    data_input_filename=${input_image}
                data_output_filename=${structure}_gradients_${num_gradients}.nii.gz
                mask_filename=GMmask.nii
                data_roi_filename=${structure}.nii
                #output_eval_filename=${outputName}_eval_31.txt
                #output_emode_filename=${outputName}_emode_31.txt
                
                if [[ ${fwhm} -gt 0 ]]; then
                    if [[ ${filter} == "TRUE" ]]; then
                        python $pipedir/connectopic_laplacian.py ${output_folder}/${data_input_filename} ${output_folder}/${data_roi_filename} \
                                                        ${output_folder}/${mask_filename} -o ${output_folder}/${data_output_filename} \
                                                        -N ${num_gradients} --smoothing ${fwhm} \
                                                        --filter
                    else
                        python $pipedir/connectopic_laplacian.py ${output_folder}/${data_input_filename} ${output_folder}/${data_roi_filename} \
                                                        ${output_folder}/${mask_filename} -o ${output_folder}/${data_output_filename} \
                                                        -N ${num_gradients} --smoothing ${fwhm}
                    fi
                elif [[ ${filter} == "TRUE" ]]; then
                    python $pipedir/connectopic_laplacian.py ${output_folder}/${data_input_filename} ${output_folder}/${data_roi_filename} \
                                                    ${output_folder}/${mask_filename} -o ${output_folder}/${data_output_filename} \
                                                    -N ${num_gradients} --filter
                else
                    python $pipedir/connectopic_laplacian.py ${output_folder}/${data_input_filename} ${output_folder}/${data_roi_filename} \
                                                    ${output_folder}/${mask_filename} -o ${output_folder}/${data_output_filename} \
                                                    -N ${num_gradients}
                fi
            done
            
        done
        
    done
}

# Invoke the main function to get things started
main "$@"