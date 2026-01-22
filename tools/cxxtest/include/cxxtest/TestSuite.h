#ifndef CXXTEST_TESTSUITE_H
#define CXXTEST_TESTSUITE_H

#include <cstddef>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace CxxTest {

class TestSuite {
public:
    virtual ~TestSuite() = default;
};

class TestAbort : public std::exception {
public:
    const char *what() const noexcept override { return "Test aborted"; }
};

struct TestCase {
    std::string suiteName;
    std::string testName;
    std::function<void()> func;
};

class TestRegistry {
public:
    static TestRegistry &instance() {
        static TestRegistry registry;
        return registry;
    }

    void add(const std::string &suiteName, const std::string &testName,
             const std::function<void()> &func) {
        _tests.push_back({suiteName, testName, func});
    }

    const std::vector<TestCase> &tests() const { return _tests; }

private:
    std::vector<TestCase> _tests;
};

class TestState {
public:
    static TestState &instance() {
        static TestState state;
        return state;
    }

    void startTest(const std::string &suite, const std::string &name) {
        _currentSuite = suite;
        _currentTest = name;
        _failuresBefore = _failures;
    }

    void fail(const std::string &message) {
        ++_failures;
        std::cerr << "[  FAILED  ] " << _currentSuite << "." << _currentTest
                  << ": " << message << std::endl;
    }

    bool currentTestFailed() const { return _failures > _failuresBefore; }

    std::size_t failures() const { return _failures; }

private:
    std::string _currentSuite;
    std::string _currentTest;
    std::size_t _failures = 0;
    std::size_t _failuresBefore = 0;
};

inline void assertTrue(bool value, const std::string &message, bool fatal) {
    if (!value) {
        TestState::instance().fail(message);
        if (fatal) {
            throw TestAbort();
        }
    }
}

template <typename T, typename U>
inline void assertEquals(const T &lhs, const U &rhs, const std::string &message,
                         bool fatal) {
    if (!(lhs == rhs)) {
        std::ostringstream oss;
        oss << message << " (lhs=" << lhs << ", rhs=" << rhs << ")";
        assertTrue(false, oss.str(), fatal);
    }
}

template <typename T, typename U>
inline void assertNotEquals(const T &lhs, const U &rhs, const std::string &message,
                            bool fatal) {
    if (lhs == rhs) {
        std::ostringstream oss;
        oss << message << " (lhs=" << lhs << ", rhs=" << rhs << ")";
        assertTrue(false, oss.str(), fatal);
    }
}

template <typename T, typename U>
inline void assertLess(const T &lhs, const U &rhs, const std::string &message,
                       bool fatal) {
    if (!(lhs < rhs)) {
        std::ostringstream oss;
        oss << message << " (lhs=" << lhs << ", rhs=" << rhs << ")";
        assertTrue(false, oss.str(), fatal);
    }
}

template <typename T, typename U>
inline void assertLessEqual(const T &lhs, const U &rhs, const std::string &message,
                            bool fatal) {
    if (!(lhs <= rhs)) {
        std::ostringstream oss;
        oss << message << " (lhs=" << lhs << ", rhs=" << rhs << ")";
        assertTrue(false, oss.str(), fatal);
    }
}

template <typename T, typename U>
inline void assertGreater(const T &lhs, const U &rhs, const std::string &message,
                          bool fatal) {
    if (!(lhs > rhs)) {
        std::ostringstream oss;
        oss << message << " (lhs=" << lhs << ", rhs=" << rhs << ")";
        assertTrue(false, oss.str(), fatal);
    }
}

template <typename T, typename U>
inline void assertGreaterEqual(const T &lhs, const U &rhs,
                               const std::string &message, bool fatal) {
    if (!(lhs >= rhs)) {
        std::ostringstream oss;
        oss << message << " (lhs=" << lhs << ", rhs=" << rhs << ")";
        assertTrue(false, oss.str(), fatal);
    }
}

template <typename Func>
inline void assertNoThrow(Func func, const std::string &message, bool fatal) {
    try {
        func();
    } catch (const std::exception &ex) {
        std::ostringstream oss;
        oss << message << " (exception=" << ex.what() << ")";
        assertTrue(false, oss.str(), fatal);
    } catch (...) {
        std::ostringstream oss;
        oss << message << " (unknown exception)";
        assertTrue(false, oss.str(), fatal);
    }
}

int runAllTests();

} // namespace CxxTest

#define TS_ASSERT(expr) \
    ::CxxTest::assertTrue(static_cast<bool>(expr), #expr, false)

#define TS_ASSERT_EQUALS(lhs, rhs) \
    ::CxxTest::assertEquals((lhs), (rhs), #lhs " == " #rhs, false)

#define TS_ASSERT_DIFFERS(lhs, rhs) \
    ::CxxTest::assertNotEquals((lhs), (rhs), #lhs " != " #rhs, false)

#define TS_ASSERT_LESS_THAN(lhs, rhs) \
    ::CxxTest::assertLess((lhs), (rhs), #lhs " < " #rhs, false)

#define TS_ASSERT_LESS_THAN_EQUALS(lhs, rhs) \
    ::CxxTest::assertLessEqual((lhs), (rhs), #lhs " <= " #rhs, false)

#define TS_ASSERT_LESS_THAN_OR_EQUALS(lhs, rhs) \
    ::CxxTest::assertLessEqual((lhs), (rhs), #lhs " <= " #rhs, false)

#define TS_ASSERT_GREATER_THAN(lhs, rhs) \
    ::CxxTest::assertGreater((lhs), (rhs), #lhs " > " #rhs, false)

#define TS_ASSERT_GREATER_THAN_EQUALS(lhs, rhs) \
    ::CxxTest::assertGreaterEqual((lhs), (rhs), #lhs " >= " #rhs, false)

#define TS_ASSERT_GREATER_THAN_OR_EQUALS(lhs, rhs) \
    ::CxxTest::assertGreaterEqual((lhs), (rhs), #lhs " >= " #rhs, false)

#define TS_ASSERT_DELTA(lhs, rhs, delta) \
    ::CxxTest::assertTrue(((lhs) >= (rhs) - (delta)) && ((lhs) <= (rhs) + (delta)), \
                          #lhs " ~= " #rhs, false)

#define TS_ASSERT_THROWS_NOTHING(expr) \
    ::CxxTest::assertNoThrow([&]() { expr; }, #expr " throws nothing", false)

#endif // CXXTEST_TESTSUITE_H
