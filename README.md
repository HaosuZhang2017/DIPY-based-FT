#DIPY-based-FT
DIPY-based fiber tracking
###
Data analysis package preparation, all the tools are availabel online.
1. DIPY: https://dipy.org/index.html
2. ANTs: https://stnava.github.io/ANTs/
3. Numpy: https://numpy.org/
4. Nibabel: https://github.com/nipy/nibabel
5. HD-BET: https://github.com/MIC-DKFZ/HD-BET
6.

For device hardware requirements, please refer to the installation requirements for different software and python toolkits.

data preparation:
All the data are in NIFTI for the current scripts.
1. T1-weighted images, ideally T1-enhanced images, data format NIFTI
2. Results of TMS language function positioning, data format NIFTI
3. DTI data, here is for images in 32 directions, data format NIFTI

The preprocessing steps follow the numbers on the script.
1. Peel off the scalp of T1
2. Align the standard image to the T1 image
3. Align the image results obtained by stimulation to the T1 image
4. DTI data preparation
5. Fiber tracking，using deterministic FT, DIPY offers the probabilistic process, too.
   thresholde setting: it differs depending on your date.
7. region identification:
   (1) The surgeon is familiar with the anatomy and can identify the area where the anchor point is located with the naked eye. It is best to have two surgeons who can compare their results.
   (2) Identification through scripts, all stimulation points, but the results still need to be checked by the surgeon.

Certainly, here is a summary of the measures recommended for enhancing the reliability of data results in fiber tracking:
1. Varying FA Thresholds: 
Analyzing fiber tracking (FT) results using different fractional anisotropy (FA) thresholds to assess the impact on the outcomes.
2. Fiber Length Threshold: Excluding Ufibers (U-fibers =< 30 mm) from the analysis due to uncertainties regarding their function and the reliability of associated algorithms.
3. Testing Different Construction Models: Experimenting with different DTI models, including Constrained Spherical Deconvolution (CSD), Robust Unbiased Model-Based Spherical Deconvolution (RUMBA-SD), etc., as data from different labs or hospitals may perform differently with various models.
Recommending manual inspection of tractography results to establish a reliable analysis path based on the performance of different model combinations.
4. Deterministic vs. Probabilistic Tractography: Noting that deterministic tractography may have higher specificity but lower sensitivity compared to probabilistic tractography.
6. Gray and White Matter Segmentation: Using brain templates to segment gray matter, white matter, and cerebrospinal fluid. Adjusting segmentation thresholds as needed for pathological conditions, mass effect, or post-surgical cavities.
    Indeed, pathological conditions, mass effects from lesions, and post-surgical cavities can pose challenges to the successful segmentation of brain regions. The adjustment of segmentation thresholds based on specific conditions is crucial to address these challenges effectively.
    1) Pathological Conditions (e.g., Edema, Hemorrhage): When dealing with images containing pathological features like edema or hemorrhage, it is essential to consider their impact on segmentation. These conditions can lead to altered tissue properties and may require adjustments to segmentation thresholds to accurately delineate regions of interest.
    2) Mass Effect on Normal Brain Tissue: Lesions or masses can exert pressure on surrounding normal brain tissue, causing displacement or deformation. In such cases, segmentation algorithms may need to account for this mass effect by adjusting thresholds or incorporating additional preprocessing steps.
    3) Post-Surgical Cavities: Post-surgical cavities within the brain can introduce voids or irregularities in the imaging data. Segmenting regions in the vicinity of these cavities may require specialized approaches or manual adjustments to ensure accurate results.
    The use of the median_otsu segmentation method (https://docs.dipy.org/stable/examples_built/preprocessing/brain_extraction_dwi.html#sphx-glr-examples-built-preprocessing-brain-extraction-dwi-py), is a good starting point. However, it is essential to adapt and fine-tune the segmentation process according to the specific characteristics of the data and the research objectives. Flexibility in adjusting segmentation thresholds and techniques is key to obtaining reliable results, particularly in the presence of atypical brain conditions.


Before applying graph theory analysis, it is essential to construct the necessary matrices for analysis. Currently, we are using a connectivity matrix based on segmented regions, which is essentially a positive matrix. 
  For instance, when utilizing AAL90 (Automated Anatomical Labeling with 90 regions) segmentation, the brain is divided into 90 regions. As a result, the constructed matrix is of size 90x90. The values on the diagonal of this matrix represent the connectivity of each region to itself, which, in our study, is considered as non-existent (i.e., these values are set to 0).
This matrix serves as the foundation for subsequent graph theory-based analyses, providing a structured representation of connectivity between specific brain regions. It is important to ensure the accuracy and appropriateness of this matrix, as it plays a crucial role in understanding network properties and relationships within the brain's functional or structural connectivity.


Regarding the construction of the connectome based on fiber tracking results:
The construction of the connectivity matrix based on the results of structural connections. The matrix is typically a positive matrix representing connectivity between regions of interest.
1. Binary Connectome: Describing the creation of a binary connectivity matrix by thresholding the FT results, where regions are marked as 0 or 1 to construct the matrix for subsequent graph theory analysis.
2. Weighted Connectome: Mentioning the alternative approach of using a weighted connectivity matrix, where fiber quantities represent the strength of connections between regions. Emphasizing that this approach has some debate regarding its applicability.
These measures are intended to ensure the reliability and robustness of data results in fiber tracking and connectome construction. Adjustments and considerations for different scenarios are highlighted to assist researchers in their analysis.



纤维追踪的算法非常重要，当然通过很多方法可以尽可能的提高结果的可信度。我们推荐使用以下不同的措施来加强数据结果。
1. 使用不同的FA阈值，对不同阈值下的FT结果进行分析。
2. 使用纤维长度阈值，将Ufiber排除在外，应为目前我们还没确定Ufiber的作用和相关算法的可靠性。
3. 使用不同的模型进行测试：
此过程中涉及的model有 Constrained Spherical Deconvolution (CSD)，Robust Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD) reconstruction model， etc。
不同实验室活着医院的数据具有不同的参数，可能对于不同的models有不同的表现。我们建议对tractography的结果进行人工检查。需要使用者根据自己数据在不同model组合之后的分析表现来构建可靠的分析路径。
5. deterministic tractography可能对于纤维追踪的敏感性低于Probablistic tractography，但是目前我们的经验是deterministic tractography的特异性高，也就是出现假阳性的结果比较少。
6. 根据脑模板对灰质和白质以及脑脊液进行分割，通过白质的mask将所有的纤维追踪结果限制在白质区域。
   但是其中会面对的问题：
   （1）当要处理的图像中有水肿出血等病理表现
   （2）当病灶对正常脑组织出现推挤
   （3）当脑组织内有手术后的空腔
  这些问题是会干扰成功的分割，要根据不同的情况调整分割的阈值。
我们通常使用的分割方法来自median_otsu(https://docs.dipy.org/stable/examples_built/preprocessing/brain_extraction_dwi.html#sphx-glr-examples-built-preprocessing-brain-extraction-dwi-py)。使用过程中要根据设计情况调整数据。


The contruction of the connectome is based on the fiber tracking results regarding structural connections.
在使用图论分析之前，我们要构建分析所需要的矩阵，目前而言我们的矩阵的是根据连接区域构建的正矩阵。例如在使用AAL90分割区域的时候，我们将脑区分割为90个，那么构建的正矩阵是90*90，对角线区域的数值代表了该区域和自己的连接，当然在本研究中，我们并不考虑这些区域自我连接的纤维，那么这些区域的数值为0.

有两种建立网络连接的方式。
1. 二元化FT结果来构建连接矩阵。由于目前DIPY进行FT的结果是不同区域之间的纤维数量，然而目前对于纤维数量和实际神经纤维数量是不能对等的，在我们自己的邮件中倾向于使用二分法来定义这下纤维，并分别标记为0或1，从而以此来构建矩阵，便后续的图论轮分析。
2. 加权的连接矩阵。与上述方案不同的是加权矩阵的分析，使用纤维数量代表不同区域之间的连接强度，但是这个立论的观点是由争议的，但是可以作为研究的补充方案，辅助研究者的判断。



The scripts for 5 kinds of centralities was applied: 
The analysis were based ont the liberay from NetworkX， We have attached links and the conceptions, as follow:
1. Degree Centrality (DC): Degree centrality measures the number of direct connections or edges that a node has in a network. Nodes with a higher degree of centrality are more connected to other nodes and are considered more central in terms of connectivity.
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.degree_centrality.html 
2. Closeness Centrality (CC): Closeness centrality assesses how close a node is to all other nodes in the network. Nodes with higher closeness centrality are more central because they can quickly reach other nodes in the network.
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
3. Betweenness Centrality (BC): Betweenness centrality quantifies the extent to which a node lies on the shortest paths between other nodes in the network. Nodes with high betweenness centrality act as bridges and are critical for maintaining network connectivity.
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html
4. Eigenvector Centrality (EC): Eigenvector centrality assigns importance to nodes based on their connections to other highly central nodes. Nodes connected to other nodes with high centrality themselves receive higher eigenvector centrality scores.
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.eigenvector_centrality.html
5. PageRank Centrality (PC): PageRank centrality, inspired by Google's PageRank algorithm, assigns importance to nodes based on both their connections and the importance of nodes they are connected to. It considers the quality of connections, not just quantity.
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html





###
