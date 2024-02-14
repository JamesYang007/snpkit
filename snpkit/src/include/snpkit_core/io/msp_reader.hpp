#pragma once
#include <cstdio>
#include <cstdlib>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace snpkit_core {
namespace io {

class MSPReader
{
public:
    using string_t = std::string;
    using file_unique_ptr_t = std::unique_ptr<
        std::FILE, 
        std::function<void(std::FILE*)>
    >;

    const string_t _filename;
    std::unordered_map<string_t, int8_t> _ancestry_map;
    std::vector<string_t> _haplotype_IDs;
    std::vector<string_t> _sample_IDs;
    std::vector<int32_t> _chm;
    std::vector<int32_t> _pos;
    std::vector<double> _gpos;
    std::vector<int32_t> _n_snps;
    std::vector<int8_t> _lai;

protected:
    static auto fopen_safe(
        const char* filename,
        const char* mode
    )
    {
        file_unique_ptr_t file_ptr(
            std::fopen(filename, mode),
            [](std::FILE* fp) { std::fclose(fp); }
        );
        auto fp = file_ptr.get();
        if (!fp) {
            throw std::runtime_error("Cannot open file " + std::string(filename));
        }
        return file_ptr;
    }

    static auto getline(
        std::vector<char>& buffer,
        std::FILE* fp
    )
    {
        assert(buffer.size() >= 2);

        size_t total_read = 0;
        auto buff_ptr = buffer.data();
        auto next_size = buffer.size();
        while (1) {
            // read a chunk of the line
            if (std::fgets(buff_ptr, next_size, fp) == nullptr) {
                // some bad error happened!
                if (std::ferror(fp)) {
                    throw std::runtime_error("Bad fgets!");
                }
                // otherwise, EOF hit, so we set buffer to be empty.
                buff_ptr[0] = 0;
            }

            const bool is_last_chunk = (
                // FILE is EOF or
                std::feof(fp) || 
                // less than count-1 read, which means newline must have been found
                (buff_ptr[next_size-2] == 0) ||
                // if count-1 read, the only way to be last chunk is if it ended with newline
                (buff_ptr[next_size-2] == '\n')
            );

            if (is_last_chunk) {
                size_t n_read = 0;
                for (; n_read < next_size && buff_ptr[n_read]; ++n_read);

                total_read += n_read;

                break;
            } else {
                size_t n_read = next_size - 1;

                total_read += n_read;
            }

            buffer.pop_back(); // remove null-terminator
            size_t old_size = buffer.size();
            next_size = old_size + 1;
            buffer.resize(old_size + next_size);
            buff_ptr = buffer.data() + old_size;
        }

        return total_read;
    }

    template <class T>
    static auto fast_atoi(
        const char* ptr,
        char delimiter,
        T& out
    )
    {
        const char* const old_ptr = ptr;
        out = 0;
        while (*ptr != delimiter) {
            out = out * 10 + (*ptr++ - '0');
        }
        return ptr - old_ptr;
    }

    static auto token_size(
        const char* ptr,
        char delimiter
    )
    {
        size_t s = 0;
        for (; ptr[s] != delimiter; ++s);
        return s;
    }

public:   
    MSPReader(
        const string_t& filename
    ):
        _filename(filename)
    {}

    void read(
        size_t max_rows,
        const string_t& delimiter_str,
        const std::vector<int>& hap_ids_indices,
        size_t buffer_size,
        size_t n_rows_hint,
        size_t n_threads
    )
    {
        if (delimiter_str.size() != 1) {
            throw std::runtime_error("delimiter must be length 1.");
        }
        if (buffer_size < 2) {
            throw std::runtime_error("buffer_size must be >= 2.");
        }
        if (n_threads < 1) {
            throw std::runtime_error("n_threads must be >= 2.");
        }

        if (max_rows == 0) return;

        _ancestry_map.clear();

        char delimiter = delimiter_str[0];
        auto file_ptr = fopen_safe(_filename.data(), "r");
        auto fp = file_ptr.get();
        
        // buffer to store each line of the file
        size_t n_read;
        std::vector<char> buffer(buffer_size); // about 4mil entries

        // read first line (ancestry information)
        // TODO: original code checks if ancestry mapping exists or not. We assume it always exists.
        n_read = getline(buffer, fp); 
        {
            constexpr const char first_line_prefix[] = "#Subpopulation order/codes: ";
            size_t idx = sizeof(first_line_prefix) - 1;

            if (string_t(buffer.data(), idx) != first_line_prefix) {
                throw std::runtime_error("First line does not contain prefix \"#Subpopulation order/codes: \"");
            }

            if (buffer[n_read-1] != '\n') {
                throw std::runtime_error("First line does not end with '\n'.");
            }
            buffer[n_read-1] = delimiter; // change the newline to delimiter

            while (idx < n_read) {
                // read ancestry key
                const auto key_size = token_size(buffer.data() + idx, '=');
                string_t key(buffer.data() + idx, key_size);
                idx += key_size + 1;

                // read ancestry value
                int8_t value;
                idx += fast_atoi(buffer.data() + idx, delimiter, value) + 1;

                // register key-value pair
                _ancestry_map[std::move(key)] = value;
            }
        }

        if (max_rows == 1) return;

        _haplotype_IDs.clear();
        _sample_IDs.clear();

        const bool all_haps = hap_ids_indices.size() == 0;

        // read second line (header information)
        n_read = getline(buffer, fp);
        {
            std::array<const char*, 6> initial_headers = {{
                "#chm",
                "spos",
                "epos",
                "sgpos",
                "egpos",
                "n snps"
            }};

            // check initial headers match
            size_t idx = 0;
            for (size_t i = 0; i < initial_headers.size(); ++i) {
                const auto size = token_size(buffer.data() + idx, delimiter);
                if (string_t(buffer.data() + idx, size) != initial_headers[i]) {
                    throw std::runtime_error("Unexpected initial header: " + string_t(initial_headers[i]));
                }
                idx += size + 1;
            }

            // check that last read character is newline, then change to delimiter for convenience
            if (buffer[n_read - 1] != '\n') {
                throw std::runtime_error("Second line does not end with '\n'.");
            }
            buffer[n_read - 1] = delimiter; // change the newline to delimiter

            // add haplotype IDs (subsetted)
            size_t count = 0;
            size_t subset_idx = 0;
            while ((idx < n_read) && (all_haps || subset_idx < hap_ids_indices.size())) {
                const auto size = token_size(buffer.data() + idx, delimiter);
                if (all_haps || count == hap_ids_indices[subset_idx]) {
                    _haplotype_IDs.emplace_back(
                        buffer.data() + idx,
                        size
                    );
                    ++subset_idx;
                }
                idx += size + 1;
                ++count;
            }
        }

        // populate sample_IDs
        if (_haplotype_IDs.size() % 2 != 0) {
            throw std::runtime_error("Haplotype IDs size is not even.");
        }
        _sample_IDs.resize(_haplotype_IDs.size() / 2);
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int i = 0; i < _sample_IDs.size(); ++i) {
            const auto& hap_id = _haplotype_IDs[2 * i];
            _sample_IDs[i] = hap_id.substr(0, hap_id.size()-2); // remove ".0" or ".1"
        }

        if (max_rows == 2) return;

        _chm.clear();
        _pos.clear();
        _gpos.clear();
        _n_snps.clear();
        _lai.clear();

        size_t n_haps = _haplotype_IDs.size();

        _chm.reserve(n_rows_hint);
        _pos.reserve(2 * n_rows_hint);
        _gpos.reserve(2 * n_rows_hint);
        _n_snps.reserve(n_rows_hint);
        _lai.reserve(n_rows_hint * n_haps);

        std::vector<std::vector<char>> buffers(n_threads);
        std::vector<size_t> n_reads(n_threads);
        for (auto& buffer : buffers) buffer.resize(buffer_size);

        size_t chm_idx = 0;
        size_t pos_idx = 0;
        size_t gpos_idx = 0;
        size_t n_snps_idx = 0;
        size_t lai_idx = 0;
        size_t n_rows = 0;

        while (!std::feof(fp) && (n_rows < max_rows-2)) {
            // populate many lines
            size_t n_lines = 0;
            while (!std::feof(fp) && (n_lines < buffers.size()) && (n_rows+n_lines < max_rows-2)) {
                n_reads[n_lines] = getline(buffers[n_lines], fp);
                if (std::feof(fp)) {
                    if (n_reads[n_lines]) {
                        throw std::runtime_error("EOF should imply that current line is empty.");
                    }
                    break;
                }
                ++n_lines;
            }
            for (size_t i = 0; i < n_lines; ++i) {
                buffers[i][n_reads[i]-1] = delimiter; // change newline to delimiter
            }

            // allocate enough space
            chm_idx = _chm.size();
            pos_idx = _pos.size();
            gpos_idx = _gpos.size();
            n_snps_idx = _n_snps.size();
            lai_idx = _lai.size();
            n_rows += n_lines;
            _chm.resize(chm_idx + n_lines);
            _pos.resize(pos_idx + 2 * n_lines);
            _gpos.resize(gpos_idx + 2 * n_lines);
            _n_snps.resize(n_snps_idx + n_lines);
            _lai.resize(lai_idx + n_lines * n_haps);

            // batch process LAI in parallel
            #pragma omp parallel for schedule(static) num_threads(n_threads)
            for (int t = 0; t < n_lines; ++t) {
                size_t idx = 0;
                for (int i = 0; i < 6; ++i) {
                    const auto size = token_size(buffers[t].data() + idx, delimiter);
                    idx += size + 1;
                }
                buffers[t][idx-1] = 0;
                std::sscanf(
                    buffers[t].data(),
                    "%d%d%d%lf%lf%d",
                    &_chm[chm_idx + t],
                    &_pos[pos_idx + 2 * t],
                    &_pos[pos_idx + 2 * t + 1],
                    &_gpos[gpos_idx + 2 * t],
                    &_gpos[gpos_idx + 2 * t + 1],
                    &_n_snps[n_snps_idx + t]
                );

                size_t count = 0;
                size_t subset_idx = 0;
                while ((idx < n_reads[t]) && (all_haps || subset_idx < hap_ids_indices.size())) {
                    size_t size;
                    if (all_haps || count == hap_ids_indices[subset_idx]) {
                        size = fast_atoi(
                            buffers[t].data() + idx,
                            delimiter,
                            _lai[lai_idx + n_haps * t + subset_idx]
                        );
                        ++subset_idx;
                    } else {
                        size = token_size(buffers[t].data() + idx, delimiter);
                    }
                    idx += size + 1;
                    ++count;
                }
            }
        }
    }
};

} // namespace io
} // namespace snpkit_core