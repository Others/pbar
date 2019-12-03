/**
 * @license
 * Copyright Google Inc. All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.io/license
 */
import { global } from './global';
export function ngDevModeResetPerfCounters() {
    var locationString = typeof location !== 'undefined' ? location.toString() : '';
    var newCounters = {
        namedConstructors: locationString.indexOf('ngDevMode=namedConstructors') != -1,
        firstTemplatePass: 0,
        tNode: 0,
        tView: 0,
        rendererCreateTextNode: 0,
        rendererSetText: 0,
        rendererCreateElement: 0,
        rendererAddEventListener: 0,
        rendererSetAttribute: 0,
        rendererRemoveAttribute: 0,
        rendererSetProperty: 0,
        rendererSetClassName: 0,
        rendererAddClass: 0,
        rendererRemoveClass: 0,
        rendererSetStyle: 0,
        rendererRemoveStyle: 0,
        rendererDestroy: 0,
        rendererDestroyNode: 0,
        rendererMoveNode: 0,
        rendererRemoveNode: 0,
        rendererAppendChild: 0,
        rendererInsertBefore: 0,
        rendererCreateComment: 0,
        styleMap: 0,
        styleMapCacheMiss: 0,
        classMap: 0,
        classMapCacheMiss: 0,
        styleProp: 0,
        stylePropCacheMiss: 0,
        classProp: 0,
        classPropCacheMiss: 0,
        flushStyling: 0,
        classesApplied: 0,
        stylesApplied: 0,
        stylingWritePersistedState: 0,
        stylingReadPersistedState: 0,
    };
    // Make sure to refer to ngDevMode as ['ngDevMode'] for closure.
    var allowNgDevModeTrue = locationString.indexOf('ngDevMode=false') === -1;
    global['ngDevMode'] = allowNgDevModeTrue && newCounters;
    return newCounters;
}
/**
 * This checks to see if the `ngDevMode` has been set. If yes,
 * then we honor it, otherwise we default to dev mode with additional checks.
 *
 * The idea is that unless we are doing production build where we explicitly
 * set `ngDevMode == false` we should be helping the developer by providing
 * as much early warning and errors as possible.
 *
 * NOTE: changes to the `ngDevMode` name must be synced with `compiler-cli/src/tooling.ts`.
 */
if (typeof ngDevMode === 'undefined' || ngDevMode) {
    ngDevModeResetPerfCounters();
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibmdfZGV2X21vZGUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi8uLi8uLi8uLi8uLi8uLi9wYWNrYWdlcy9jb3JlL3NyYy91dGlsL25nX2Rldl9tb2RlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7R0FNRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxVQUFVLENBQUM7QUE0Q2hDLE1BQU0sVUFBVSwwQkFBMEI7SUFDeEMsSUFBTSxjQUFjLEdBQUcsT0FBTyxRQUFRLEtBQUssV0FBVyxDQUFDLENBQUMsQ0FBQyxRQUFRLENBQUMsUUFBUSxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztJQUNsRixJQUFNLFdBQVcsR0FBMEI7UUFDekMsaUJBQWlCLEVBQUUsY0FBYyxDQUFDLE9BQU8sQ0FBQyw2QkFBNkIsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUM5RSxpQkFBaUIsRUFBRSxDQUFDO1FBQ3BCLEtBQUssRUFBRSxDQUFDO1FBQ1IsS0FBSyxFQUFFLENBQUM7UUFDUixzQkFBc0IsRUFBRSxDQUFDO1FBQ3pCLGVBQWUsRUFBRSxDQUFDO1FBQ2xCLHFCQUFxQixFQUFFLENBQUM7UUFDeEIsd0JBQXdCLEVBQUUsQ0FBQztRQUMzQixvQkFBb0IsRUFBRSxDQUFDO1FBQ3ZCLHVCQUF1QixFQUFFLENBQUM7UUFDMUIsbUJBQW1CLEVBQUUsQ0FBQztRQUN0QixvQkFBb0IsRUFBRSxDQUFDO1FBQ3ZCLGdCQUFnQixFQUFFLENBQUM7UUFDbkIsbUJBQW1CLEVBQUUsQ0FBQztRQUN0QixnQkFBZ0IsRUFBRSxDQUFDO1FBQ25CLG1CQUFtQixFQUFFLENBQUM7UUFDdEIsZUFBZSxFQUFFLENBQUM7UUFDbEIsbUJBQW1CLEVBQUUsQ0FBQztRQUN0QixnQkFBZ0IsRUFBRSxDQUFDO1FBQ25CLGtCQUFrQixFQUFFLENBQUM7UUFDckIsbUJBQW1CLEVBQUUsQ0FBQztRQUN0QixvQkFBb0IsRUFBRSxDQUFDO1FBQ3ZCLHFCQUFxQixFQUFFLENBQUM7UUFDeEIsUUFBUSxFQUFFLENBQUM7UUFDWCxpQkFBaUIsRUFBRSxDQUFDO1FBQ3BCLFFBQVEsRUFBRSxDQUFDO1FBQ1gsaUJBQWlCLEVBQUUsQ0FBQztRQUNwQixTQUFTLEVBQUUsQ0FBQztRQUNaLGtCQUFrQixFQUFFLENBQUM7UUFDckIsU0FBUyxFQUFFLENBQUM7UUFDWixrQkFBa0IsRUFBRSxDQUFDO1FBQ3JCLFlBQVksRUFBRSxDQUFDO1FBQ2YsY0FBYyxFQUFFLENBQUM7UUFDakIsYUFBYSxFQUFFLENBQUM7UUFDaEIsMEJBQTBCLEVBQUUsQ0FBQztRQUM3Qix5QkFBeUIsRUFBRSxDQUFDO0tBQzdCLENBQUM7SUFFRixnRUFBZ0U7SUFDaEUsSUFBTSxrQkFBa0IsR0FBRyxjQUFjLENBQUMsT0FBTyxDQUFDLGlCQUFpQixDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7SUFDNUUsTUFBTSxDQUFDLFdBQVcsQ0FBQyxHQUFHLGtCQUFrQixJQUFJLFdBQVcsQ0FBQztJQUN4RCxPQUFPLFdBQVcsQ0FBQztBQUNyQixDQUFDO0FBRUQ7Ozs7Ozs7OztHQVNHO0FBQ0gsSUFBSSxPQUFPLFNBQVMsS0FBSyxXQUFXLElBQUksU0FBUyxFQUFFO0lBQ2pELDBCQUEwQixFQUFFLENBQUM7Q0FDOUIiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGUgbGljZW5zZSB0aGF0IGNhbiBiZVxuICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICovXG5cbmltcG9ydCB7Z2xvYmFsfSBmcm9tICcuL2dsb2JhbCc7XG5cbmRlY2xhcmUgZ2xvYmFsIHtcbiAgY29uc3QgbmdEZXZNb2RlOiBudWxsfE5nRGV2TW9kZVBlcmZDb3VudGVycztcbiAgaW50ZXJmYWNlIE5nRGV2TW9kZVBlcmZDb3VudGVycyB7XG4gICAgbmFtZWRDb25zdHJ1Y3RvcnM6IGJvb2xlYW47XG4gICAgZmlyc3RUZW1wbGF0ZVBhc3M6IG51bWJlcjtcbiAgICB0Tm9kZTogbnVtYmVyO1xuICAgIHRWaWV3OiBudW1iZXI7XG4gICAgcmVuZGVyZXJDcmVhdGVUZXh0Tm9kZTogbnVtYmVyO1xuICAgIHJlbmRlcmVyU2V0VGV4dDogbnVtYmVyO1xuICAgIHJlbmRlcmVyQ3JlYXRlRWxlbWVudDogbnVtYmVyO1xuICAgIHJlbmRlcmVyQWRkRXZlbnRMaXN0ZW5lcjogbnVtYmVyO1xuICAgIHJlbmRlcmVyU2V0QXR0cmlidXRlOiBudW1iZXI7XG4gICAgcmVuZGVyZXJSZW1vdmVBdHRyaWJ1dGU6IG51bWJlcjtcbiAgICByZW5kZXJlclNldFByb3BlcnR5OiBudW1iZXI7XG4gICAgcmVuZGVyZXJTZXRDbGFzc05hbWU6IG51bWJlcjtcbiAgICByZW5kZXJlckFkZENsYXNzOiBudW1iZXI7XG4gICAgcmVuZGVyZXJSZW1vdmVDbGFzczogbnVtYmVyO1xuICAgIHJlbmRlcmVyU2V0U3R5bGU6IG51bWJlcjtcbiAgICByZW5kZXJlclJlbW92ZVN0eWxlOiBudW1iZXI7XG4gICAgcmVuZGVyZXJEZXN0cm95OiBudW1iZXI7XG4gICAgcmVuZGVyZXJEZXN0cm95Tm9kZTogbnVtYmVyO1xuICAgIHJlbmRlcmVyTW92ZU5vZGU6IG51bWJlcjtcbiAgICByZW5kZXJlclJlbW92ZU5vZGU6IG51bWJlcjtcbiAgICByZW5kZXJlckFwcGVuZENoaWxkOiBudW1iZXI7XG4gICAgcmVuZGVyZXJJbnNlcnRCZWZvcmU6IG51bWJlcjtcbiAgICByZW5kZXJlckNyZWF0ZUNvbW1lbnQ6IG51bWJlcjtcbiAgICBzdHlsZU1hcDogbnVtYmVyO1xuICAgIHN0eWxlTWFwQ2FjaGVNaXNzOiBudW1iZXI7XG4gICAgY2xhc3NNYXA6IG51bWJlcjtcbiAgICBjbGFzc01hcENhY2hlTWlzczogbnVtYmVyO1xuICAgIHN0eWxlUHJvcDogbnVtYmVyO1xuICAgIHN0eWxlUHJvcENhY2hlTWlzczogbnVtYmVyO1xuICAgIGNsYXNzUHJvcDogbnVtYmVyO1xuICAgIGNsYXNzUHJvcENhY2hlTWlzczogbnVtYmVyO1xuICAgIGZsdXNoU3R5bGluZzogbnVtYmVyO1xuICAgIGNsYXNzZXNBcHBsaWVkOiBudW1iZXI7XG4gICAgc3R5bGVzQXBwbGllZDogbnVtYmVyO1xuICAgIHN0eWxpbmdXcml0ZVBlcnNpc3RlZFN0YXRlOiBudW1iZXI7XG4gICAgc3R5bGluZ1JlYWRQZXJzaXN0ZWRTdGF0ZTogbnVtYmVyO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBuZ0Rldk1vZGVSZXNldFBlcmZDb3VudGVycygpOiBOZ0Rldk1vZGVQZXJmQ291bnRlcnMge1xuICBjb25zdCBsb2NhdGlvblN0cmluZyA9IHR5cGVvZiBsb2NhdGlvbiAhPT0gJ3VuZGVmaW5lZCcgPyBsb2NhdGlvbi50b1N0cmluZygpIDogJyc7XG4gIGNvbnN0IG5ld0NvdW50ZXJzOiBOZ0Rldk1vZGVQZXJmQ291bnRlcnMgPSB7XG4gICAgbmFtZWRDb25zdHJ1Y3RvcnM6IGxvY2F0aW9uU3RyaW5nLmluZGV4T2YoJ25nRGV2TW9kZT1uYW1lZENvbnN0cnVjdG9ycycpICE9IC0xLFxuICAgIGZpcnN0VGVtcGxhdGVQYXNzOiAwLFxuICAgIHROb2RlOiAwLFxuICAgIHRWaWV3OiAwLFxuICAgIHJlbmRlcmVyQ3JlYXRlVGV4dE5vZGU6IDAsXG4gICAgcmVuZGVyZXJTZXRUZXh0OiAwLFxuICAgIHJlbmRlcmVyQ3JlYXRlRWxlbWVudDogMCxcbiAgICByZW5kZXJlckFkZEV2ZW50TGlzdGVuZXI6IDAsXG4gICAgcmVuZGVyZXJTZXRBdHRyaWJ1dGU6IDAsXG4gICAgcmVuZGVyZXJSZW1vdmVBdHRyaWJ1dGU6IDAsXG4gICAgcmVuZGVyZXJTZXRQcm9wZXJ0eTogMCxcbiAgICByZW5kZXJlclNldENsYXNzTmFtZTogMCxcbiAgICByZW5kZXJlckFkZENsYXNzOiAwLFxuICAgIHJlbmRlcmVyUmVtb3ZlQ2xhc3M6IDAsXG4gICAgcmVuZGVyZXJTZXRTdHlsZTogMCxcbiAgICByZW5kZXJlclJlbW92ZVN0eWxlOiAwLFxuICAgIHJlbmRlcmVyRGVzdHJveTogMCxcbiAgICByZW5kZXJlckRlc3Ryb3lOb2RlOiAwLFxuICAgIHJlbmRlcmVyTW92ZU5vZGU6IDAsXG4gICAgcmVuZGVyZXJSZW1vdmVOb2RlOiAwLFxuICAgIHJlbmRlcmVyQXBwZW5kQ2hpbGQ6IDAsXG4gICAgcmVuZGVyZXJJbnNlcnRCZWZvcmU6IDAsXG4gICAgcmVuZGVyZXJDcmVhdGVDb21tZW50OiAwLFxuICAgIHN0eWxlTWFwOiAwLFxuICAgIHN0eWxlTWFwQ2FjaGVNaXNzOiAwLFxuICAgIGNsYXNzTWFwOiAwLFxuICAgIGNsYXNzTWFwQ2FjaGVNaXNzOiAwLFxuICAgIHN0eWxlUHJvcDogMCxcbiAgICBzdHlsZVByb3BDYWNoZU1pc3M6IDAsXG4gICAgY2xhc3NQcm9wOiAwLFxuICAgIGNsYXNzUHJvcENhY2hlTWlzczogMCxcbiAgICBmbHVzaFN0eWxpbmc6IDAsXG4gICAgY2xhc3Nlc0FwcGxpZWQ6IDAsXG4gICAgc3R5bGVzQXBwbGllZDogMCxcbiAgICBzdHlsaW5nV3JpdGVQZXJzaXN0ZWRTdGF0ZTogMCxcbiAgICBzdHlsaW5nUmVhZFBlcnNpc3RlZFN0YXRlOiAwLFxuICB9O1xuXG4gIC8vIE1ha2Ugc3VyZSB0byByZWZlciB0byBuZ0Rldk1vZGUgYXMgWyduZ0Rldk1vZGUnXSBmb3IgY2xvc3VyZS5cbiAgY29uc3QgYWxsb3dOZ0Rldk1vZGVUcnVlID0gbG9jYXRpb25TdHJpbmcuaW5kZXhPZignbmdEZXZNb2RlPWZhbHNlJykgPT09IC0xO1xuICBnbG9iYWxbJ25nRGV2TW9kZSddID0gYWxsb3dOZ0Rldk1vZGVUcnVlICYmIG5ld0NvdW50ZXJzO1xuICByZXR1cm4gbmV3Q291bnRlcnM7XG59XG5cbi8qKlxuICogVGhpcyBjaGVja3MgdG8gc2VlIGlmIHRoZSBgbmdEZXZNb2RlYCBoYXMgYmVlbiBzZXQuIElmIHllcyxcbiAqIHRoZW4gd2UgaG9ub3IgaXQsIG90aGVyd2lzZSB3ZSBkZWZhdWx0IHRvIGRldiBtb2RlIHdpdGggYWRkaXRpb25hbCBjaGVja3MuXG4gKlxuICogVGhlIGlkZWEgaXMgdGhhdCB1bmxlc3Mgd2UgYXJlIGRvaW5nIHByb2R1Y3Rpb24gYnVpbGQgd2hlcmUgd2UgZXhwbGljaXRseVxuICogc2V0IGBuZ0Rldk1vZGUgPT0gZmFsc2VgIHdlIHNob3VsZCBiZSBoZWxwaW5nIHRoZSBkZXZlbG9wZXIgYnkgcHJvdmlkaW5nXG4gKiBhcyBtdWNoIGVhcmx5IHdhcm5pbmcgYW5kIGVycm9ycyBhcyBwb3NzaWJsZS5cbiAqXG4gKiBOT1RFOiBjaGFuZ2VzIHRvIHRoZSBgbmdEZXZNb2RlYCBuYW1lIG11c3QgYmUgc3luY2VkIHdpdGggYGNvbXBpbGVyLWNsaS9zcmMvdG9vbGluZy50c2AuXG4gKi9cbmlmICh0eXBlb2YgbmdEZXZNb2RlID09PSAndW5kZWZpbmVkJyB8fCBuZ0Rldk1vZGUpIHtcbiAgbmdEZXZNb2RlUmVzZXRQZXJmQ291bnRlcnMoKTtcbn1cbiJdfQ==