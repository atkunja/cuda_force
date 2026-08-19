#include "bench_common.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

using namespace cudaforge::bench;

// The JSON writer produces every benchmark result file in this repository. A
// malformed emitter would not fail loudly — it would write files that a parser
// rejects days later, after the machine that produced them is gone. These tests
// exist so the output format is verified rather than assumed.

namespace {

/// Minimal structural validation: balanced braces and brackets, quotes closed,
/// no trailing comma before a closer. Enough to catch every way this emitter
/// could realistically break, without pulling in a JSON parser as a test-only
/// dependency.
bool structurally_valid(const std::string& text) {
    int braces = 0;
    int brackets = 0;
    bool in_string = false;
    char previous = '\0';

    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (c == '"' && previous != '\\') {
                in_string = false;
            }
            previous = c;
            continue;
        }
        switch (c) {
            case '"':
                in_string = true;
                break;
            case '{':
                ++braces;
                break;
            case '}':
                if (--braces < 0) {
                    return false;
                }
                break;
            case '[':
                ++brackets;
                break;
            case ']':
                if (--brackets < 0) {
                    return false;
                }
                break;
            case ',': {
                // A comma immediately before a closer is the classic malformed
                // JSON this emitter's separator logic exists to avoid.
                std::size_t j = i + 1;
                while (j < text.size() && std::isspace(static_cast<unsigned char>(text[j]))) {
                    ++j;
                }
                if (j < text.size() && (text[j] == '}' || text[j] == ']')) {
                    return false;
                }
                break;
            }
            default:
                break;
        }
        previous = c;
    }
    return braces == 0 && brackets == 0 && !in_string;
}

}  // namespace

TEST_CASE("an empty object is valid", "[bench][json]") {
    std::ostringstream out;
    JsonWriter writer(out);
    writer.begin_object();
    writer.end_object();
    writer.finish();

    REQUIRE(structurally_valid(out.str()));
}

TEST_CASE("fields are comma separated without a trailing comma", "[bench][json]") {
    std::ostringstream out;
    JsonWriter writer(out);
    writer.begin_object();
    writer.field("name", std::string("value"));
    writer.field("count", std::uint64_t{42});
    writer.field("ratio", 0.5);
    writer.end_object();
    writer.finish();

    const std::string text = out.str();
    INFO(text);
    REQUIRE(structurally_valid(text));
    REQUIRE(text.find("\"name\": \"value\"") != std::string::npos);
    REQUIRE(text.find("\"count\": 42") != std::string::npos);
    REQUIRE(text.find("\"ratio\": 0.5") != std::string::npos);
}

TEST_CASE("nested arrays of objects stay balanced", "[bench][json]") {
    // The shape every benchmark actually emits.
    std::ostringstream out;
    JsonWriter writer(out);
    writer.begin_object();
    writer.field("benchmark", std::string("example"));
    writer.begin_array("cases");
    for (int i = 0; i < 3; ++i) {
        writer.array_element_begin();
        writer.field("index", static_cast<std::uint64_t>(i));
        writer.field("seconds", 0.125 * i);
        writer.array_element_end();
    }
    writer.end_array();
    writer.end_object();
    writer.finish();

    INFO(out.str());
    REQUIRE(structurally_valid(out.str()));
}

TEST_CASE("an empty array is valid", "[bench][json]") {
    std::ostringstream out;
    JsonWriter writer(out);
    writer.begin_object();
    writer.begin_array("cases");
    writer.end_array();
    writer.end_object();
    writer.finish();

    REQUIRE(structurally_valid(out.str()));
}

TEST_CASE("output ends with a newline", "[bench][json]") {
    // Result files are appended to logs and read line by line; a missing
    // terminator concatenates the last field with whatever follows.
    std::ostringstream out;
    JsonWriter writer(out);
    writer.begin_object();
    writer.field("x", 1.0);
    writer.end_object();
    writer.finish();

    REQUIRE(out.str().back() == '\n');
}

TEST_CASE("percentiles interpolate between samples", "[bench][stats]") {
    std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0};
    REQUIRE(percentile(samples, 0.0) == 1.0);
    REQUIRE(percentile(samples, 0.5) == 3.0);
    REQUIRE(percentile(samples, 1.0) == 5.0);
    // 0.25 * (5 - 1) = 1.0, exactly the second sample.
    REQUIRE(percentile(samples, 0.25) == 2.0);
}

TEST_CASE("percentiles sort the input", "[bench][stats]") {
    std::vector<double> shuffled = {5.0, 1.0, 4.0, 2.0, 3.0};
    REQUIRE(percentile(shuffled, 0.5) == 3.0);
}

TEST_CASE("percentiles of an empty sample set are zero", "[bench][stats]") {
    std::vector<double> empty;
    REQUIRE(percentile(empty, 0.5) == 0.0);
}

TEST_CASE("percentiles are monotonic", "[bench][stats]") {
    std::vector<double> samples;
    samples.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        samples.push_back(static_cast<double>(i) * 0.5);
    }
    REQUIRE(percentile(samples, 0.5) <= percentile(samples, 0.9));
    REQUIRE(percentile(samples, 0.9) <= percentile(samples, 0.99));
    REQUIRE(percentile(samples, 0.99) <= percentile(samples, 1.0));
}

TEST_CASE("mean matches the arithmetic average", "[bench][stats]") {
    REQUIRE(mean({1.0, 2.0, 3.0, 4.0}) == 2.5);
    REQUIRE(mean({}) == 0.0);
}

TEST_CASE("the timer measures elapsed time", "[bench][timer]") {
    Timer timer;
    timer.start();
    // Only a lower bound: an upper bound would be flaky under load.
    while (timer.elapsed_seconds() < 0.002) {
    }
    REQUIRE(timer.elapsed_seconds() >= 0.002);
    REQUIRE(timer.elapsed_ms() >= 2.0);
}

TEST_CASE("the timer never reports a negative duration", "[bench][timer]") {
    // steady_clock rather than high_resolution_clock, which on some
    // implementations aliases system_clock and can step backwards.
    Timer timer;
    timer.start();
    REQUIRE(timer.elapsed_seconds() >= 0.0);
}
