// Recompile the current public fixture against the reference SDK's own ABI.
// No solver code is copied or reimplemented by this comparison executable.
extern int snippetChainmailHeadlessMain(int, const char* const*);
int main(int argc, char** argv)
{
    return snippetChainmailHeadlessMain(argc, argv);
}
