#include <cxxtest/TestSuite.h>
#include <iostream>

namespace CxxTest {

int runAllTests() {
    const auto &tests = TestRegistry::instance().tests();
    std::size_t total = tests.size();
    std::size_t failed = 0;

    std::cout << "[==========] Running " << total << " tests." << std::endl;

    for (const auto &test : tests) {
        TestState::instance().startTest(test.suiteName, test.testName);
        std::cout << "[ RUN      ] " << test.suiteName << "." << test.testName << std::endl;
        try {
            test.func();
        } catch (const TestAbort &) {
            // Failure already recorded
        } catch (const std::exception &ex) {
            TestState::instance().fail(std::string("Unhandled exception: ") + ex.what());
        } catch (...) {
            TestState::instance().fail("Unhandled unknown exception");
        }

        if (TestState::instance().currentTestFailed()) {
            ++failed;
            std::cout << "[  FAILED  ] " << test.suiteName << "." << test.testName << std::endl;
        } else {
            std::cout << "[       OK ] " << test.suiteName << "." << test.testName << std::endl;
        }
    }

    std::cout << "[==========] " << total << " tests ran. " << failed
              << " failed." << std::endl;
    return failed == 0 ? 0 : 1;
}

} // namespace CxxTest
