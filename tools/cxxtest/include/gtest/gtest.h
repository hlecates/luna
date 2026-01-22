#ifndef GTEST_GTEST_H
#define GTEST_GTEST_H

#include <cxxtest/TestSuite.h>
#include <string>

namespace testing {

class Test {
public:
    virtual ~Test() = default;
    virtual void SetUp() {}
    virtual void TearDown() {}
};

} // namespace testing

namespace detail {

inline std::string makeTestName(const std::string &suite, const std::string &name) {
    return suite + "." + name;
}

class TestRegistrar {
public:
    TestRegistrar(const std::string &suite, const std::string &name,
                  const std::function<void()> &func) {
        CxxTest::TestRegistry::instance().add(suite, name, func);
    }
};

} // namespace detail

#define TEST(test_suite_name, test_name)                                  \
    static void test_suite_name##_##test_name##_impl();                   \
    static detail::TestRegistrar                                           \
        test_suite_name##_##test_name##_registrar(#test_suite_name, #test_name, \
            []() { test_suite_name##_##test_name##_impl(); });            \
    static void test_suite_name##_##test_name##_impl()

#define TEST_F(test_fixture, test_name)                                   \
    class test_fixture##_##test_name##_Test : public test_fixture {       \
    public:                                                               \
        void TestBody();                                                  \
        void Run() {                                                      \
            this->SetUp();                                                \
            try {                                                         \
                TestBody();                                               \
            } catch (...) {                                               \
                this->TearDown();                                         \
                throw;                                                    \
            }                                                             \
            this->TearDown();                                             \
        }                                                                 \
    };                                                                    \
    static detail::TestRegistrar                                          \
        test_fixture##_##test_name##_registrar(#test_fixture, #test_name, \
            []() {                                                        \
                test_fixture##_##test_name##_Test instance;               \
                instance.Run();                                           \
            });                                                           \
    void test_fixture##_##test_name##_Test::TestBody()

#define EXPECT_TRUE(expr) \
    ::CxxTest::assertTrue(static_cast<bool>(expr), #expr, false)

#define ASSERT_TRUE(expr) \
    ::CxxTest::assertTrue(static_cast<bool>(expr), #expr, true)

#define EXPECT_FALSE(expr) \
    ::CxxTest::assertTrue(!(expr), "!" #expr, false)

#define ASSERT_FALSE(expr) \
    ::CxxTest::assertTrue(!(expr), "!" #expr, true)

#define EXPECT_EQ(lhs, rhs) \
    ::CxxTest::assertEquals((lhs), (rhs), #lhs " == " #rhs, false)

#define ASSERT_EQ(lhs, rhs) \
    ::CxxTest::assertEquals((lhs), (rhs), #lhs " == " #rhs, true)

#define EXPECT_NE(lhs, rhs) \
    ::CxxTest::assertNotEquals((lhs), (rhs), #lhs " != " #rhs, false)

#define ASSERT_NE(lhs, rhs) \
    ::CxxTest::assertNotEquals((lhs), (rhs), #lhs " != " #rhs, true)

#define EXPECT_LT(lhs, rhs) \
    ::CxxTest::assertLess((lhs), (rhs), #lhs " < " #rhs, false)

#define ASSERT_LT(lhs, rhs) \
    ::CxxTest::assertLess((lhs), (rhs), #lhs " < " #rhs, true)

#define EXPECT_LE(lhs, rhs) \
    ::CxxTest::assertLessEqual((lhs), (rhs), #lhs " <= " #rhs, false)

#define ASSERT_LE(lhs, rhs) \
    ::CxxTest::assertLessEqual((lhs), (rhs), #lhs " <= " #rhs, true)

#define EXPECT_GT(lhs, rhs) \
    ::CxxTest::assertGreater((lhs), (rhs), #lhs " > " #rhs, false)

#define ASSERT_GT(lhs, rhs) \
    ::CxxTest::assertGreater((lhs), (rhs), #lhs " > " #rhs, true)

#define EXPECT_GE(lhs, rhs) \
    ::CxxTest::assertGreaterEqual((lhs), (rhs), #lhs " >= " #rhs, false)

#define ASSERT_GE(lhs, rhs) \
    ::CxxTest::assertGreaterEqual((lhs), (rhs), #lhs " >= " #rhs, true)

#define EXPECT_NO_THROW(statement) \
    ::CxxTest::assertNoThrow([&]() { statement; }, #statement " throws nothing", false)

#define ASSERT_NO_THROW(statement) \
    ::CxxTest::assertNoThrow([&]() { statement; }, #statement " throws nothing", true)

#endif // GTEST_GTEST_H
