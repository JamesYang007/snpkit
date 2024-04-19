#pragma once
#include <snpkit_core/util/types.hpp>

namespace snpkit_core {

template <class T>
util::colarr_type<T>
to_sample_major(
    const Eigen::Ref<const util::rowarr_type<T>>& m,
    size_t n_threads
)
{
    if (m.cols() % 2 != 0) {
        throw std::runtime_error("Number of columns is not even!");
    }
    const auto n_snps = m.rows();
    const auto n_samples = m.cols() / 2;
    util::colarr_type<T> out(n_samples, 2 * n_snps);

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (size_t j = 0; j < n_snps; ++j) {
        const Eigen::Map<const util::rowarr_type<T>> slice(
            m.row(j).data(),
            n_samples,
            2
        );
        out.col(2 * j) = slice.col(0);
        out.col(2 * j + 1) = slice.col(1);
    }

    return out;
}

template <class T>
util::colarr_type<T>
calldata_sum(
    const Eigen::Ref<const util::colarr_type<T>>& m,
    size_t n_threads
) 
{
    const auto n = m.rows();
    const auto p = m.cols() / 2;
    util::colarr_type<T> out(n, p);

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (size_t j = 0; j < p; ++j) {
        out.col(j) = m.col(2*j) + m.col(2*j+1);
    }

    return out;
}

template <class T, class V>
std::tuple<util::rowvec_type<V>, util::rowvec_type<size_t>> 
column_mean(
    const Eigen::Ref<const util::colarr_type<T>>& m,
    size_t n_threads
)
{
    util::rowvec_type<V> out(m.cols());
    util::rowvec_type<size_t> missing(m.cols());

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (size_t j = 0; j < m.cols(); ++j) {
        int sum = 0;
        int n_miss = 0;
        for (size_t k = 0; k < m.rows(); ++k) {
            if (m(k,j) > 0) {
                sum += m(k,j);
            } else {
                ++n_miss;
            }
        }
        out[j] = static_cast<V>(sum) / (m.rows() - n_miss);
        missing[j] = n_miss;
    }

    return std::make_tuple(out, missing);
}

template <class ValueType, class IntType>
util::colmat_type<ValueType> calldata_subset_rows_cols(
    const Eigen::Ref<const util::colmat_type<ValueType>>& calldata, 
    const Eigen::Ref<const util::rowvec_type<IntType>>& row_indices,
    const Eigen::Ref<const util::rowvec_type<IntType>>& col_indices,
    size_t n_threads
)
{
    util::colmat_type<ValueType> out(row_indices.size(), 2 * col_indices.size());
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (size_t j = 0; j < col_indices.size(); ++j) {
        const auto jj = col_indices[j];
        for (size_t k = 0; k < 2; ++k) {
            const auto m_j = calldata.col(2 * jj + k);
            auto out_j = out.col(2 * j + k);
            for (size_t i = 0; i < row_indices.size(); ++i) {
                out_j[i] = m_j[row_indices[i]];
            }
        }
    }
    return out;
}

} // namespace snpkit_core