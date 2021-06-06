#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#include <random>
#include <opencv2/opencv.hpp>
#include "./nanoflann/include/nanoflann.hpp" 

using namespace nanoflann;

struct PointCloud
{
	std::vector<cv::Mat>  pts;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		if (dim == 0) return pts[idx].at<float>(0,0);
		else if (dim == 1) return pts[idx].at<float>(1,0);
		else return pts[idx].at<float>(2,0);
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

typedef struct Trafo{
    cv::Mat rot;
    float scale;
    cv::Mat offset1,offset2;
} Trafo;

typedef struct Correspondence {
    int data_index;
    int model_index;
    float sqr_dist;
} Correspondence;

PointCloud loadPointCloud(char* filename, float stddev) {
    
    std::ifstream file(filename);
    if(!file.is_open()) {
        std::cerr << "No such file: " << filename << std::endl;
        exit(1);
    }
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0.0, stddev);
    PointCloud points;
    float coord;
    while(file >> coord) {
        cv::Mat point(3,1, CV_32F);
        point.at<float>(0,0) = coord + dist(generator);
        file >> coord;
        point.at<float>(1,0) = coord + dist(generator);
        file >> coord;
        point.at<float>(2,0) = coord + dist(generator);
        points.pts.push_back(point.clone());
    }

    return points;
}

void writePointCloud(PointCloud& cloud, const std::string& filename) {
    std::ofstream outfile(filename);

    for(int i = 0; i < cloud.pts.size(); ++i) {
        outfile << cloud.pts[i].at<float>(0,0) << " " << cloud.pts[i].at<float>(1,0) << " " << cloud.pts[i].at<float>(2,0) << std::endl;
    }
}

std::vector<Correspondence> findNearestNeighbours(PointCloud& data, PointCloud& model) {
    std::vector<Correspondence> correspondences;

    // construct a kd-tree index:
	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<float, PointCloud > ,
		PointCloud,
		3 /* dim */
		> my_kd_tree_t;   

    my_kd_tree_t   index(3 /*dim*/, model, KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );
    index.buildIndex(); 

    for(int i = 0; i < data.pts.size(); ++i) {
        float query_pt[3] = {data.pts[i].at<float>(0,0), data.pts[i].at<float>(1,0), data.pts[i].at<float>(2,0)};

        // do a knn search
        const size_t num_results = 1;
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(&ret_index, &out_dist_sqr );
        index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

        // std::cout << "knnSearch(nn="<<num_results<<"): \n";
	    // std::cout << "ret_index=" << ret_index << " out_dist_sqr=" << out_dist_sqr << std::endl;
        Correspondence c;
        c.data_index = i;
        c.model_index = ret_index;
        c.sqr_dist = out_dist_sqr;
        // sqr_dists.push_back(out_dist_sqr);
        // indices.push_back(ret_index);
        correspondences.push_back(c);
    }

    return correspondences;
}

Trafo calculateMotion(PointCloud& data, PointCloud& model, const std::vector<Correspondence>& correspondences) {
    Trafo ret;

    //Subtract offsets
    cv::Mat offset1 = (cv::Mat_<float>(3,1) << 0, 0, 0);
    cv::Mat offset2 = (cv::Mat_<float>(3,1) << 0, 0, 0);

    for(int i = 0; i < correspondences.size(); ++i) {
        offset2 += data.pts[correspondences[i].data_index];
        offset1 += model.pts[correspondences[i].model_index];
    }

    offset1 /= data.pts.size();
    offset2 /= data.pts.size();

    ret.offset1 = offset1;
    ret.offset2 = offset2;

    cv::Mat H = cv::Mat::zeros(3,3,CV_32F);
    for(int i = 0; i < correspondences.size(); ++i) {
        cv::Mat v2 = data.pts[correspondences[i].data_index];
        cv::Mat v1 = model.pts[correspondences[i].model_index];

        float x1 = v1.at<float>(0,0) - offset1.at<float>(0,0);
        float y1 = v1.at<float>(1,0) - offset1.at<float>(1,0);
        float z1 = v1.at<float>(2,0) - offset1.at<float>(2,0);

        float x2 = v2.at<float>(0,0) - offset2.at<float>(0,0);
        float y2 = v2.at<float>(1,0) - offset2.at<float>(1,0);
        float z2 = v2.at<float>(2,0) - offset2.at<float>(2,0);

        H.at<float>(0,0)+=x2*x1;
		H.at<float>(0,1)+=x2*y1;
		H.at<float>(0,2)+=x2*z1;

		H.at<float>(1,0)+=y2*x1;
		H.at<float>(1,1)+=y2*y1;
		H.at<float>(1,2)+=y2*z1;

		H.at<float>(2,0)+=z2*x1;
		H.at<float>(2,1)+=z2*y1;
		H.at<float>(2,2)+=z2*z1;
    }

    cv::Mat w(3,3,CV_32F);
	cv::Mat u(3,3,CV_32F);
	cv::Mat vt(3,3,CV_32F);

	cv::SVD::compute(H,w,u, vt);

	cv::Mat rot=vt.t()*u.t();
	ret.rot=rot;

    return ret;
}

void calculateAlignmentError(cv::Mat& data_rotation, cv::Mat& data_translation, cv::Mat& rotation, cv::Mat& translation) {
    cv::Mat rotvec(3,1,CV_32F) ;
    cv::Rodrigues(data_rotation*rotation, rotvec);
    
    float rotation_error = 0;
    float translation_error = 0;
    for(int i = 0; i < 3; ++i) {
        rotation_error += rotvec.at<float>(i,0) * rotvec.at<float>(i,0);
        translation_error += (data_translation.at<float>(i,0) + translation.at<float>(i,0)) * (data_translation.at<float>(i,0) + translation.at<float>(i,0));
    }
    std::cout << "Rotation error: " << std::sqrt(rotation_error) << std::endl;
    std::cout << "Translation error: " << std::sqrt(translation_error) << std::endl;
}

void ICP(PointCloud& data, PointCloud& model, int iteration, cv::Mat& data_rotation, cv::Mat& data_translation) {

    float mse_old = 0;
    cv::Mat rotation = cv::Mat::eye(3,3,CV_32F);
    cv::Mat translation = cv::Mat::zeros(3,1,CV_32F);

    for(int i = 0; i < iteration; ++i) {
        std::vector<Correspondence> correspondences = findNearestNeighbours(data, model);
        Trafo motion = calculateMotion(data, model, correspondences);

        // float mse = std::accumulate(sqr_dists.begin(), sqr_dists.end(), 0.0) / indices.size();
        float mse = 0;
        for(int i = 0; i < correspondences.size(); ++i) {
            mse += correspondences[i].sqr_dist;
        }
        mse /= correspondences.size();

        

        if(std::abs(mse_old - mse) < 0.0001){
            std::cout << "ICP iteration: " << i+1 << " MSE: " << mse << std::endl;
            break;
        } else {
            std::cout << "ICP iteration: " << i+1 << " MSE: " << mse << "            \r" << std::flush;
        }
        mse_old = mse;

        // apply motion
        // offset1 -> model offset, offset2 -> data offset
        for(int i = 0; i < data.pts.size(); ++i) {
            data.pts[i] -= motion.offset2;
            data.pts[i] = motion.rot * data.pts[i];
        }

        for(int i = 0; i < model.pts.size(); ++i) {
            model.pts[i] -= motion.offset1;
        }
        rotation *= motion.rot;
        translation += motion.offset1 - motion.offset2;
    }
    std::cout << "Rotation:" << std::endl;
    std::cout << rotation << std::endl;
    std::cout << "Translation: " << std::endl;
    std::cout << translation << std::endl;
    calculateAlignmentError(data_rotation, data_translation, rotation, translation);
}

void TrICP(PointCloud& data, PointCloud& model, int iteration, float min_overlap, cv::Mat& data_rotation, cv::Mat& data_translation) {

    float mse_old = 0;
    int N_po = std::round(min_overlap * data.pts.size());
    cv::Mat rotation = cv::Mat::eye(3,3,CV_32F);
    cv::Mat translation = cv::Mat::zeros(3,1,CV_32F);

    for(int i = 0; i < iteration; ++i) {
        std::vector<Correspondence> all_correspondences = findNearestNeighbours(data, model);

        std::sort(all_correspondences.begin(), all_correspondences.end(), [](Correspondence a, Correspondence b) {
            return a.sqr_dist < b.sqr_dist;   
        });

        std::vector<Correspondence> correspondences(all_correspondences.begin(), all_correspondences.begin()+N_po);

        Trafo motion = calculateMotion(data, model, correspondences);

        float mse = 0;
        for(int i = 0; i < correspondences.size(); ++i) {
            mse += correspondences[i].sqr_dist;
        }
        mse /= correspondences.size();

        if(std::abs(mse_old - mse) < 0.0001){
            std::cout << "TrICP iteration: " << i+1 << " MSE: " << mse << std::endl;
            break;
        } else {
            std::cout << "TrICP iteration: " << i+1 << " MSE: " << mse << "            \r" << std::flush;
        }
        mse_old = mse;

        // apply motion
        // offset1 -> model offset, offset2 -> data offset
        for(int i = 0; i < data.pts.size(); ++i) {
            data.pts[i] -= motion.offset2;
            data.pts[i] = motion.rot * data.pts[i];
        }

        for(int i = 0; i < model.pts.size(); ++i) {
            model.pts[i] -= motion.offset1;
        }
        rotation *= motion.rot;
        translation += motion.offset1 - motion.offset2;
        
    }
    std::cout << "Rotation:" << std::endl;
    std::cout << rotation << std::endl;
    std::cout << "Translation: " << std::endl;
    std::cout << translation << std::endl;
    calculateAlignmentError(data_rotation, data_translation, rotation, translation);
}

int main(int argc, char** argv) {

    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " DATA MODEL MIN_OVERLAP MAX_ITERATION OUTPUT_FILENAME NOISE_STDDEV DATA_TRANSFORMATION_FILE" << std::endl;
        return 1;
    }

    int iteration = std::stoi(argv[4]);
    float min_overlap = std::stof(argv[3]);
    std::string output_filename = argv[5];
    float stddev = std::stof(argv[6]);

    cv::Mat data_rotation = cv::Mat::eye(3,3,CV_32F);
    cv::Mat data_translation = cv::Mat::zeros(3,1,CV_32F);
    std::ifstream transform_file(argv[7]);
    if(!transform_file.is_open()) {
        std::cerr << "Transform file: " << argv[7] << " does not exist!" << std::endl;
        return 1;
    }
    for(int i = 0; i < 3; ++i) {
        float value;
        transform_file >> value;
        data_translation.at<float>(i,0) = value;
    }
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            float value;
            transform_file >> value;
            data_rotation.at<float>(i,j) = value;
        }
    }
    std::cout << data_rotation << std::endl << data_translation << std::endl;
    

    std::cout << "Loading pointclouds...\r"; 
    PointCloud data = loadPointCloud(argv[1], stddev);
    PointCloud model = loadPointCloud(argv[2], stddev);
    PointCloud data_TrICP;
    PointCloud model_TrICP;
    for(int i = 0; i < data.pts.size(); ++i) {
        data_TrICP.pts.push_back(data.pts[i].clone());
    }
    for(int i = 0; i < model.pts.size(); ++i) {
        model_TrICP.pts.push_back(model.pts[i].clone());
    }
    std::cout << "Loadaing pointclouds... Done." << std::endl;

    std::cout << "Calculating ICP:" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    ICP(data, model, iteration, data_rotation, data_translation);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sec = t2 - t1;
    std::cout << "ICP runtime: " << sec.count() << " seconds." << std::endl;
    std::cout << "Writing results to " << output_filename << "_(data/model)_ICP.xyz...\r"; 
    writePointCloud(data, output_filename + "_data_ICP.xyz");
    writePointCloud(model, output_filename + "_model_ICP.xyz");
    std::cout << "Writing results to " << output_filename << "_(data/model)_ICP.xyz... Done." << std::endl; 

    

    std::cout << "Calculating TrICP:" << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    TrICP(data_TrICP, model_TrICP, iteration, min_overlap, data_rotation, data_translation);
    t2 = std::chrono::high_resolution_clock::now();
    sec = t2 - t1;
    std::cout << "TrICP runtime: " << sec.count() << " seconds." << std::endl;
    std::cout << "Writing results to " << output_filename << "_(data/model)_TrICP.xyz...\r";
    writePointCloud(data_TrICP, output_filename + "_data_TrICP.xyz");
    writePointCloud(model_TrICP, output_filename + "_model_TrICP.xyz");
    std::cout << "Writing results to " << output_filename << "_(data/model)_TrICP.xyz... Done." << std::endl; 
    return 0;
}